import argparse
import copy
import datetime
import json
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR


def load_text_token_file(file_path):
    """Load token IDs from text file (chr-encoded, UTF-8)."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Convert each character to its Unicode code point (token ID)
    token_ids = [ord(char) for char in content]
    return torch.tensor(token_ids, dtype=torch.long)


def get_batch(data, batch_size, block_size, device):
    """Generate a small batch of data of inputs x and targets y."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

def get_lr(it, warmup_steps, total_steps, max_lr, min_lr):
    # 1) Linear warmup for the first few steps
    if it < warmup_steps:
        return max_lr * it / warmup_steps
    
    # 2) If we exceed total_steps, return min_lr
    if it > total_steps:
        return min_lr
    
    # 3) In between, use cosine decay down to min_lr
    decay_ratio = (it - warmup_steps) / (total_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
    
    return min_lr + coeff * (max_lr - min_lr)


def apply_rotary_emb(x, cos, sin):
    """
    Apply rotary embeddings to x. x has shape (..., seq_len, head_size);
    cos, sin have shape (seq_len, head_size/2) and will be broadcast.
    """
    # x: (..., T, hs) -> (..., T, hs/2, 2)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    # cos, sin: (T, hs/2) -> need to broadcast to (..., T, hs/2)
    x_rotated_0 = x1 * cos - x2 * sin
    x_rotated_1 = x1 * sin + x2 * cos
    # Interleave back: (..., T, hs)
    x_out = torch.stack([x_rotated_0, x_rotated_1], dim=-1).flatten(-2)
    return x_out


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE). No learned parameters; uses buffer for inverse frequencies."""

    def __init__(self, head_size, base=10000.0):
        super().__init__()
        assert head_size % 2 == 0, "head_size must be even for RoPE"
        inv_freq = 1.0 / (base ** (torch.arange(0, head_size, 2).float() / head_size))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, positions=None):
        """
        Apply RoPE to x. x has shape (..., T, head_size).
        If positions is None, use torch.arange(T, device=x.device).
        """
        T = x.size(-2)
        device, dtype = x.device, x.dtype
        if positions is None:
            positions = torch.arange(T, device=device, dtype=dtype)
        # inv_freq: (head_size/2,) -> (1, head_size/2)
        inv_freq = self.inv_freq.to(dtype=dtype)
        freqs = positions.unsqueeze(-1) * inv_freq.unsqueeze(0)  # (T, head_size/2)
        cos = freqs.cos()
        sin = freqs.sin()
        return apply_rotary_emb(x, cos, sin)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, eval_iters, device):
    """Estimate loss on train and validation sets."""
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, batch_size, block_size, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        assert head_size % 2 == 0, "head_size must be even for RoPE"
        self.RESIDUAL_SCALE_INIT = True
        self.n_head = num_heads
        self.head_size = head_size
        self.rope = RotaryEmbedding(head_size)
        # One big linear layer for all Q, K, V
        self.qkv_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.proj.RESIDUAL_SCALE_INIT = True
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        # Batch, Time, (3 * Heads * HeadSize)
        q, k, v = self.qkv_attn(x).split(C, dim=2)
        
        # Reshape for multi-head parallel processing
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Apply rotary position embedding to Q and K
        q = self.rope(q)
        k = self.rope(k)

        # Standard Scaled Dot-Product Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        # Re-assemble heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        self.net[2].RESIDUAL_SCALE_INIT = True

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, block_size, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        assert head_size % 2 == 0, "n_embd // n_head must be even for RoPE"
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, dropout, device, label_smoothing):
        super().__init__()
        self.block_size = block_size
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.device = device
        self.label_smoothing = label_smoothing
        self.n_layer = n_layer
        self.apply(self._init_weights)
        self.lm_head.weight = self.token_embedding_table.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'RESIDUAL_SCALE_INIT'):
                std *= (2 * self.n_layer) ** -0.5 # Scaling for depth
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            std = 0.02
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers (position encoding via RoPE in attention)
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        x = tok_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)

        return logits, loss

    def generate(self, idx, max_new_tokens, block_size, temperature=1.0, top_k=None):
        # idx is (B, T) array of indices in the current context
        # Ensure we have at least one token to start with
        if idx.shape[1] == 0:
            raise ValueError("Cannot generate from empty context. Context must contain at least one token.")
        
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] / temperature # becomes (B, C), apply temperature
            # optionally crop the logits to only the top_k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


def main():
    parser = argparse.ArgumentParser(
        description="Train a GPT language model on tokenized card data."
    )
    
    # Required arguments
    parser.add_argument(
        '--train-file',
        type=str,
        required=True,
        help='Path to training data file (text format, chr-encoded token IDs)'
    )
    parser.add_argument(
        '--test-file',
        type=str,
        required=True,
        help='Path to test data file (text format, chr-encoded token IDs)'
    )
    parser.add_argument(
        '--token-map',
        type=str,
        required=True,
        help='Path to token map JSON file'
    )
    parser.add_argument(
        '--output-model',
        type=str,
        required=True,
        help='Path where trained model will be saved'
    )
    
    # Optional hyperparameters
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Batch size (default: 64)'
    )
    parser.add_argument(
        '--block-size',
        type=int,
        default=256,
        help='Maximum context length for predictions (default: 256)'
    )
    parser.add_argument(
        '--max-iters',
        type=int,
        default=5000,
        help='Maximum number of training iterations (default: 5000)'
    )
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=500,
        help='Interval for evaluating loss (default: 500)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=3e-4,
        help='Learning rate (default: 3e-4)'
    )
    parser.add_argument(
        '--eval-iters',
        type=int,
        default=200,
        help='Number of iterations for loss estimation (default: 200)'
    )
    parser.add_argument(
        '--n-embd',
        type=int,
        default=384,
        help='Embedding dimension (default: 384)'
    )
    parser.add_argument(
        '--n-head',
        type=int,
        default=6,
        help='Number of attention heads (default: 6)'
    )
    parser.add_argument(
        '--n-layer',
        type=int,
        default=6,
        help='Number of transformer layers (default: 6)'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.2,
        help='Dropout rate (default: 0.2)'
    )
    parser.add_argument(
        '--label-smoothing',
        type=float,
        default=0.1,
        help='Label smoothing rate (default: 0.1)'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.1,
        help='Weight decay rate (default: 0.1)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1337,
        help='Random seed (default: 1337)'
    )
    parser.add_argument(
        '--min-lr',
        type=float,
        default=0.0,
        help='Minimum learning rate for cosine decay scheduler (default: 0.0)'
    )
    parser.add_argument(
        '--early-stop-patience',
        type=int,
        default=5,
        help='Number of evaluation intervals to wait before early stopping (default: 5)'
    )
    parser.add_argument(
        '--warmup-steps',
        type=int,
        default=500,
        help='Number of steps for linear warmup (default: 500)'
    )
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load token map
    print(f"Loading token map from {args.token_map}...")
    with open(args.token_map, 'r') as f:
        token_map_data = json.load(f)
        token_map = token_map_data['token_map']
        decoder = {v: k for k, v in token_map.items()}
        vocab_size = len(token_map)
    print(f"Vocabulary size: {vocab_size}")
    
    # Load training and test data
    print(f"Loading training data from {args.train_file}...")
    train_data = load_text_token_file(args.train_file)
    print(f"Training data size: {len(train_data)} tokens")
    
    print(f"Loading test data from {args.test_file}...")
    val_data = load_text_token_file(args.test_file)
    print(f"Test data size: {len(val_data)} tokens")
    
    # Create model
    print("Initializing model...")
    model = GPTLanguageModel(
        vocab_size=vocab_size,
        n_embd=args.n_embd,
        block_size=args.block_size,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
        device=device,
        label_smoothing=args.label_smoothing
    )
    model = model.to(device)
    
    # Print number of parameters
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{num_params:.2f}M parameters")
    
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    
    optim_groups = [
        {'params': decay_params, 'weight_decay': args.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=args.learning_rate, betas=(0.9, 0.95))
    
    # Early stopping tracking variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    early_stopped = False
    
    # Training loop
    print(f"Starting training for {args.max_iters} iterations...")
    for iter in range(args.max_iters):
        # Evaluate loss periodically
        lr = get_lr(iter, args.warmup_steps, args.max_iters, args.learning_rate, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if iter % args.eval_interval == 0 or iter == args.max_iters - 1:
            losses = estimate_loss(
                model, train_data, val_data,
                args.batch_size, args.block_size,
                args.eval_iters, device
            )
            print(f"step {iter}, time {str(datetime.datetime.now())}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.6f}")
            
            # Early stopping logic
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
                print(f"  -> New best validation loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"  -> No improvement ({patience_counter}/{args.early_stop_patience})")
            
            # Check for early stopping
            if patience_counter >= args.early_stop_patience:
                print(f"\nEarly stopping triggered after {iter} iterations (patience: {args.early_stop_patience})")
                early_stopped = True
                break
            
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            generated_ids = model.generate(context, max_new_tokens=50, block_size=args.block_size, temperature=0.8, top_k=50)[0].tolist()
            generated_tokens = [decoder.get(token_id, f"<UNK_{token_id}>") for token_id in generated_ids]
            generated_text = ''.join(generated_tokens)
            print(generated_text)
        
        # Sample a batch of data
        xb, yb = get_batch(train_data, args.batch_size, args.block_size, device)
        
        # Evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    # Restore best model if early stopping occurred or if we have a best model state
    if best_model_state is not None:
        print(f"\nRestoring best model with validation loss: {best_val_loss:.4f}")
        model.load_state_dict(best_model_state)
        if early_stopped:
            print("(Early stopping was triggered)")
    
    # Save model
    print(f"Saving model to {args.output_model}...")
    torch.save(model, args.output_model)
    print(f"Model saved successfully to {args.output_model}")
    
    # Generate from the model
    print("\nGenerating sample output...")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_ids = model.generate(context, max_new_tokens=50, block_size=args.block_size, temperature=0.8, top_k=50)[0].tolist()
    generated_tokens = [decoder.get(token_id, f"<UNK_{token_id}>") for token_id in generated_ids]
    generated_text = ''.join(generated_tokens)
    print(generated_text)
    
    # Generate longer output to file
    print("\nGenerating longer output to more.txt...")
    generated_ids_long = model.generate(context, max_new_tokens=1000, block_size=args.block_size, temperature=0.8, top_k=50)[0].tolist()
    generated_tokens_long = [decoder.get(token_id, f"<UNK_{token_id}>") for token_id in generated_ids_long]
    generated_text_long = ''.join(generated_tokens_long)
    with open('more.txt', 'w', encoding='utf-8') as f:
        f.write(generated_text_long)
    print("Longer output saved to more.txt")


if __name__ == "__main__":
    main()
