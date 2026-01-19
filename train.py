import argparse
import datetime
import json
import torch
import torch.nn as nn
from torch.nn import functional as F


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
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, block_size, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, dropout, device):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.device = device

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, block_size):
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
            logits = logits[:, -1, :] # becomes (B, C)
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
        default=64,
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
        '--seed',
        type=int,
        default=1337,
        help='Random seed (default: 1337)'
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
        device=device
    )
    model = model.to(device)
    
    # Print number of parameters
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{num_params:.2f}M parameters")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    print(f"Starting training for {args.max_iters} iterations...")
    for iter in range(args.max_iters):
        # Evaluate loss periodically
        if iter % args.eval_interval == 0 or iter == args.max_iters - 1:
            losses = estimate_loss(
                model, train_data, val_data, 
                args.batch_size, args.block_size, 
                args.eval_iters, device
            )
            print(f"step {iter}, time {str(datetime.datetime.now())}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            generated_ids = model.generate(context, max_new_tokens=50, block_size=args.block_size)[0].tolist()
            generated_tokens = [decoder.get(token_id, f"<UNK_{token_id}>") for token_id in generated_ids]
            generated_text = ''.join(generated_tokens)
            print(generated_text)
        
        # Sample a batch of data
        xb, yb = get_batch(train_data, args.batch_size, args.block_size, device)
        
        # Evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    # Save model
    print(f"Saving model to {args.output_model}...")
    torch.save(model, args.output_model)
    print(f"Model saved successfully to {args.output_model}")
    
    # Generate from the model
    print("\nGenerating sample output...")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_ids = model.generate(context, max_new_tokens=50, block_size=args.block_size)[0].tolist()
    generated_tokens = [decoder.get(token_id, f"<UNK_{token_id}>") for token_id in generated_ids]
    generated_text = ''.join(generated_tokens)
    print(generated_text)
    
    # Generate longer output to file
    print("\nGenerating longer output to more.txt...")
    generated_ids_long = model.generate(context, max_new_tokens=1000, block_size=args.block_size)[0].tolist()
    generated_tokens_long = [decoder.get(token_id, f"<UNK_{token_id}>") for token_id in generated_ids_long]
    generated_text_long = ''.join(generated_tokens_long)
    with open('more.txt', 'w', encoding='utf-8') as f:
        f.write(generated_text_long)
    print("Longer output saved to more.txt")


if __name__ == "__main__":
    main()
