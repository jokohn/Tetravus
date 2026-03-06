import argparse
import json
import os
import random
import sys
import torch

from train import GPTLanguageModel, Block, MultiHeadAttention, Head, FeedFoward, RotaryEmbedding
from special_tokens import begin_card_token, end_card_token
from card import Card
from token_stream import TokenStream
from tokenizers.tokenize_name import detokenize_name
from tokenizers.tokenize_oracle_text import detokenize_oracle_text
from tokenizers.tokenize_mana_cost import detokenize_mana_cost
from tokenizers.tokenize_type_line import detokenize_type_line
from tokenizers.tokenize_simple_card_fields import (
    detokenize_release_year, detokenize_rarity, detokenize_set_name,
    detokenize_power, detokenize_toughness, detokenize_loyalty
)
from special_tokens import (
    begin_name_token, end_name_token, begin_oracle_text_token,
    end_oracle_text_token, begin_mana_cost_token, end_mana_cost_token,
    begin_type_line_token, end_type_line_token,
    fim_begin_token, fim_end_token, sentinel_tokens,
)
from fim_utils import build_fim_prompt_for_inference

def load_model_and_token_map(model_path, token_map_path, device):
    """
    Load a trained model and token map from files.
    
    Args:
        model_path: Path to saved model file (.pt)
        token_map_path: Path to token map JSON file
        device: Device to load model on ('cuda' or 'cpu')
        
    Returns:
        Tuple of (model, token_map, decoder, block_size, vocab_size)
    """
    # Validate files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(token_map_path):
        raise FileNotFoundError(f"Token map file not found: {token_map_path}")
    
    # Load token map
    print(f"Loading token map from {token_map_path}...")
    with open(token_map_path, 'r') as f:
        token_map_data = json.load(f)
        token_map = token_map_data['token_map']
        decoder = {v: k for k, v in token_map.items()}
        vocab_size = len(token_map)
    print(f"Vocabulary size: {vocab_size}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = torch.load(model_path, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()  # Set to evaluation mode

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{num_params:.2f}M parameters")

    print(f"Model parameters: {model.parameters()}")
    
    # Extract block_size from model (RoPE models use model.block_size; legacy checkpoints use position_embedding_table)
    if hasattr(model, 'block_size'):
        block_size = model.block_size
    else:
        block_size = model.position_embedding_table.weight.shape[0]
    print(f"Model block size: {block_size}")
    
    return model, token_map, decoder, block_size, vocab_size


def parse_prompt_tokens(prompt_string):
    """
    Parse a prompt string by splitting on '><' while preserving angle brackets.
    
    Example:
        Input: '<card><name><name_char_G>'
        Output: ['<card>', '<name>', '<name_char_G>']
    
    Args:
        prompt_string: String containing concatenated tokens like '<card><name>...'
        
    Returns:
        List of token strings
    """
    if not prompt_string:
        return []
    
    # Split on '><' pattern
    parts = prompt_string.split('><')
    
    if len(parts) == 1:
        # No '><' found, return as single token (might already be complete)
        return [prompt_string] if prompt_string else []
    
    # Reconstruct tokens: first part gets '>' appended, middle parts get '<' prepended and '>' appended,
    # last part gets '<' prepended and '>' appended if needed
    tokens = []
    
    # First token: add '>' if not already present
    first = parts[0]
    if not first.endswith('>'):
        first += '>'
    tokens.append(first)
    
    # Middle tokens: add '<' at start and '>' at end
    for part in parts[1:-1]:
        token = '<' + part + '>'
        tokens.append(token)
    
    # Last token: add '<' at start and '>' at end if not already present
    if len(parts) > 1:
        last = parts[-1]
        if not last.startswith('<'):
            last = '<' + last
        if not last.endswith('>'):
            last += '>'
        tokens.append(last)
    
    return tokens


def initialize_context(token_map, prompt_list=None, device='cpu'):
    """
    Initialize the context tensor for generation.
    
    Args:
        token_map: Dictionary mapping token strings to token IDs
        prompt: Optional prompt string (token sequence)
        device: Device to create tensor on
        
    Returns:
        Tensor of shape (1, T) containing token IDs
    """
    if prompt_list is None:
        # Default: start with begin_card_token
        token_id = token_map.get(begin_card_token, 0)
        context = torch.tensor([[token_id]], dtype=torch.long, device=device)
    else:
        # Convert prompt token strings to token ID
        # Since tokens are unique, we can directly look up the prompt
        token_list = []
        for token in prompt_list:
            if (token_id := token_map.get(token, None)) is not None:
                token_list.append(token_id)
            else:
                print(f"Warning: Prompt '{token}' not found in token map, skipping.")
        
        # Ensure we have at least one token (default to begin_card_token if empty)
        if not token_list:
            print(f"Warning: No valid tokens in prompt. Defaulting to {begin_card_token}.")
            token_id = token_map.get(begin_card_token, 0)
            token_list = [token_id]
        
        context = torch.tensor([token_list], dtype=torch.long, device=device)
    return context


def generate_tokens(model, context, num_tokens, block_size, device):
    """
    Generate tokens using the model.
    
    Args:
        model: GPTLanguageModel instance
        context: Initial context tensor of shape (1, T)
        num_tokens: Number of new tokens to generate
        block_size: Maximum context length
        device: Device to run on
        
    Returns:
        Tuple of (full_sequence_tensor, new_tokens_tensor)
        - full_sequence_tensor: Complete sequence including context and generated tokens
        - new_tokens_tensor: Only the newly generated tokens
    """
    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=num_tokens, block_size=block_size)
    
    # Extract only the newly generated tokens (everything after the initial context)
    new_tokens = generated[0, context.shape[1]:].clone()
    
    return generated, new_tokens


def decode_tokens(token_ids, decoder):
    """
    Decode token IDs back to token strings.
    
    Args:
        token_ids: Tensor or list of token IDs
        decoder: Dictionary mapping token IDs to token strings
        
    Returns:
        List of token strings
    """
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    
    tokens = [decoder.get(token_id, f"<UNK_{token_id}>") for token_id in token_ids]
    return tokens


def format_output(tokens):
    """
    Format token list as a readable string.
    
    Args:
        tokens: List of token strings
        
    Returns:
        Formatted string
    """
    return ''.join(tokens)


def create_card_from_args(args):
    """
    Create a Card object from command-line arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Card object with provided fields, None for missing fields
    """
    return Card(
        name=args.name,
        oracle_text=args.oracle_text,
        mana_cost=args.mana_cost,
        type_line=args.type_line,
        release_year=args.release_year,
        rarity=args.rarity,
        set_name=args.set,
        power=args.power,
        toughness=args.toughness,
        loyalty=args.loyalty
    )


def _parse_chunk_into_fields(chunk_tokens, field_names, card):
    """
    Try to parse a chunk of tokens as a sequence of fields (e.g. one run).
    For each field name in order, attempt to detokenize and set on card.
    Stops on first parse failure or when stream is consumed.

    Returns:
        Number of fields successfully parsed.
    """
    if not chunk_tokens or not field_names:
        return 0
    stream = TokenStream(chunk_tokens)
    parsed = 0
    for field_name in field_names:
        if not stream.has_next():
            break
        try:
            if field_name == "name" and stream.peek() == begin_name_token:
                card.name = detokenize_name(stream)
                parsed += 1
            elif field_name == "mana_cost" and stream.peek() == begin_mana_cost_token:
                card.mana_cost = detokenize_mana_cost(stream)
                parsed += 1
            elif field_name == "type_line" and stream.peek() == begin_type_line_token:
                card.type_line = detokenize_type_line(stream)
                parsed += 1
            elif field_name == "oracle_text" and stream.peek() == begin_oracle_text_token:
                card.oracle_text = detokenize_oracle_text(stream, card_name=card.name)
                parsed += 1
            elif field_name == "release_year" and stream.peek().startswith("<release_year_"):
                card.release_year = detokenize_release_year(stream)
                parsed += 1
            elif field_name == "rarity" and stream.peek().startswith("<rarity_"):
                card.rarity = detokenize_rarity(stream)
                parsed += 1
            elif field_name == "set" and stream.peek().startswith("<set_"):
                card.set_code = detokenize_set_name(stream)
                parsed += 1
            elif field_name == "power" and stream.peek().startswith("<power_"):
                card.power = detokenize_power(stream)
                parsed += 1
            elif field_name == "toughness" and stream.peek().startswith("<toughness_"):
                card.toughness = detokenize_toughness(stream)
                parsed += 1
            elif field_name == "loyalty" and stream.peek().startswith("<loyalty_"):
                card.loyalty = detokenize_loyalty(stream)
                parsed += 1
            else:
                print(f"Unknown token: {stream.peek()}")
                break
        except (ValueError, IndexError) as e:
            print(f"Error parsing token: {e}")
            break
    return parsed


def _split_generated_by_sentinels(generated_tokens):
    """
    Split generated token list by sentinel tokens (sentinel_1, sentinel_2, ...) and </card>.
    Returns list of chunks; chunk[i] is the content between sentinel_i and sentinel_{i+1} (or </card>).
    """
    chunks = []
    current = []
    for t in generated_tokens:
        if t == end_card_token:
            if current:
                chunks.append(current)
            break
        if t in sentinel_tokens[1:]:  # skip sentinel_0 (we're already past it)
            if current:
                chunks.append(current)
            current = []
        else:
            current.append(t)
    if current:
        chunks.append(current)
    return chunks


def parse_generated_sentinel_tail(generated_tokens, runs, card):
    """
    Parse the model-generated token list (content after </FIM><sentinel_0>).
    Split by sentinel tokens into chunks; each chunk corresponds to one run of fields.
    Try to parse each chunk into the run's fields and set on card.

    Args:
        generated_tokens: List of token strings (model output after prompt)
        runs: List of lists of field names (from build_fim_prompt_for_inference)
        card: Card to update with parsed fields (modified in place)
    """
    chunks = _split_generated_by_sentinels(generated_tokens)
    print(f"Chunks: {chunks}")
    for i, run_fields in enumerate(runs):
        if i >= len(chunks):
            break
        _parse_chunk_into_fields(chunks[i], run_fields, card)


def parse_tokens_to_card(tokens, initial_card):
    """
    Parse a token stream back into a Card object.
    
    Args:
        tokens: List of token strings
        initial_card: Card object with initial values (may be partially filled)
        
    Returns:
        Card object with parsed values merged with initial values
    """
    
    # Create a copy of the initial card to avoid modifying it
    card = Card(
        name=initial_card.name,
        oracle_text=initial_card.oracle_text,
        mana_cost=initial_card.mana_cost,
        type_line=initial_card.type_line,
        release_year=initial_card.release_year,
        rarity=initial_card.rarity,
        set_name=initial_card.set_code,
        power=initial_card.power,
        toughness=initial_card.toughness,
        loyalty=initial_card.loyalty
    )
    
    # Create token stream
    stream = TokenStream(tokens)
    
    # Skip begin_card_token if present
    if stream.has_next() and stream.peek() == begin_card_token:
        stream.advance()
    
    # Parse tokens until end_card_token or end of stream
    while stream.has_next():
        current_token = stream.peek()

        # Check for end_card_token
        if current_token == end_card_token:
            stream.advance()
            return card
        # Parse name field
        elif current_token == begin_name_token:
            if card.name is None:  # Only parse if not already set
                card.name = detokenize_name(stream)
            else:
                # Skip this field since it's already set
                stream.advance()
                while stream.has_next() and stream.peek() != end_name_token:
                    stream.advance()
                if stream.has_next():
                    stream.advance()
        # Parse oracle_text field
        elif current_token == begin_oracle_text_token:
            if card.oracle_text is None:  # Only parse if not already set
                card.oracle_text = detokenize_oracle_text(stream, card_name=card.name)
            else:
                # Skip this field since it's already set
                stream.advance()
                while stream.has_next() and stream.peek() != end_oracle_text_token:
                    stream.advance()
                if stream.has_next():
                    stream.advance()
        # Parse mana_cost field
        elif current_token == begin_mana_cost_token:
            if card.mana_cost is None:  # Only parse if not already set
                card.mana_cost = detokenize_mana_cost(stream)
            else:
                # Skip this field since it's already set
                stream.advance()
                while stream.has_next() and stream.peek() != end_mana_cost_token:
                    stream.advance()
                if stream.has_next():
                    stream.advance()        
        # Parse type_line field
        elif current_token == begin_type_line_token:
            if card.type_line is None:  # Only parse if not already set
                card.type_line = detokenize_type_line(stream)
            else:
                # Skip this field since it's already set
                stream.advance()
                while stream.has_next() and stream.peek() != end_type_line_token:
                    stream.advance()
                if stream.has_next():
                    stream.advance()
        # Parse simple single-token fields
        elif current_token.startswith('<release_year_'):
            if card.release_year is None:
                card.release_year = detokenize_release_year(stream)
            else:
                stream.advance()
        elif current_token.startswith('<rarity_'):
            if card.rarity is None:
                card.rarity = detokenize_rarity(stream)
            else:
                stream.advance()
        elif current_token.startswith('<set_'):
            if card.set_code is None:
                card.set_code = detokenize_set_name(stream)
            else:
                stream.advance()
        elif current_token.startswith('<power_'):
            if card.power is None:
                card.power = detokenize_power(stream)
            else:
                stream.advance()
        elif current_token.startswith('<toughness_'):
            if card.toughness is None:
                card.toughness = detokenize_toughness(stream)
            else:
                stream.advance()
        elif current_token.startswith('<loyalty_'):
            if card.loyalty is None:
                card.loyalty = detokenize_loyalty(stream)
            else:
                stream.advance()
        else:       
        # Unknown token, skip it
            stream.advance()


def print_card(card):
    """
    Print a card in a human-readable format.
    
    Args:
        card: Card object to print
    """
    print("\n" + "="*80)
    print("GENERATED CARD")
    print("="*80)
    
    # Name
    if card.name:
        print(f"Name: {card.name}")
    else:
        print("Name: [Not generated]")
    
    # Mana Cost
    if card.mana_cost:
        print(f"Mana Cost: {card.mana_cost}")
    else:
        print("Mana Cost: [Not generated]")
    
    # Type Line
    if card.type_line:
        print(f"Type: {card.type_line}")
    else:
        print("Type: [Not generated]")
    
    # Oracle Text
    if card.oracle_text:
        print(f"\nOracle Text:")
        print(card.oracle_text)
    else:
        print("\nOracle Text: [Not generated]")
    
    # Power/Toughness (for creatures)
    if card.power is not None and card.toughness is not None:
        print(f"\nPower/Toughness: {card.power}/{card.toughness}")
    elif card.power is not None:
        print(f"\nPower: {card.power}")
    elif card.toughness is not None:
        print(f"\nToughness: {card.toughness}")
    
    # Loyalty (for planeswalkers)
    if card.loyalty is not None:
        print(f"Loyalty: {card.loyalty}")
    
    # Set and Rarity
    if card.set_code:
        print(f"\nSet: {card.set_code}")
    if card.rarity:
        print(f"Rarity: {card.rarity}")
    if card.release_year:
        print(f"Release Year: {card.release_year}")
    
    print("="*80 + "\n")


def interactive_generation_loop(model, initial_context, token_map, decoder, block_size, device, output_file=None):
    """
    Interactive loop for generating tokens.
    
    Args:
        model: GPTLanguageModel instance
        initial_context: Initial context tensor
        token_map: Dictionary mapping token strings to token IDs
        decoder: Dictionary mapping token IDs to token strings
        block_size: Maximum context length
        device: Device to run on
        output_file: Optional file path to save output
    """
    context = initial_context.clone()
    all_generated_tokens = []
    
    print("\n" + "="*80)
    print("Starting interactive generation. Type 'n' or 'q' to quit.")
    print("="*80 + "\n")
    
    try:
        while True:
            # Prompt user
            user_input = input("Generate more tokens? (y/n) or enter number of tokens: ").strip().lower()
            
            if user_input in ['n', 'q', 'no', 'quit', 'exit']:
                print("\nExiting...")
                break
            
            # Determine number of tokens to generate
            if user_input == 'y' or user_input == '':
                num_tokens = 50  # Default
            else:
                try:
                    num_tokens = int(user_input)
                    if num_tokens <= 0:
                        print("Please enter a positive number of tokens.")
                        continue
                except ValueError:
                    print("Invalid input. Please enter 'y', 'n', or a number.")
                    continue
            
            # Generate tokens
            print(f"\nGenerating {num_tokens} tokens...")
            full_sequence, new_tokens = generate_tokens(model, context, num_tokens, block_size, device)
            
            # Decode new tokens
            new_token_strings = decode_tokens(new_tokens, decoder)
            all_generated_tokens.extend(new_token_strings)
            
            # Update context for next iteration
            context = full_sequence
            
            # Format and display
            new_output = format_output(new_token_strings)
            print("\n" + "-"*80)
            print("New tokens generated:")
            print("-"*80)
            print(new_output)
            print("-"*80 + "\n")
            
            # Save to file if specified
            if output_file:
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(new_output)
                print(f"Appended to {output_file}\n")
            
            # Show current full context length
            print(f"Current context length: {context.shape[1]} tokens\n")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    
    # Print summary
    if all_generated_tokens:
        print("\n" + "="*80)
        print("Generation Summary")
        print("="*80)
        full_output = format_output(all_generated_tokens)
        print(f"Total tokens generated: {len(all_generated_tokens)}")
        print(f"\nFull generated output:\n{full_output}")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Load a trained model and generate card fields from partial card information."
    )
    
    # Required arguments
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to saved model file (.pt)'
    )
    parser.add_argument(
        '--token-map',
        type=str,
        required=True,
        help='Path to token map JSON file'
    )
    
    # Card field arguments (all optional)
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Card name'
    )
    parser.add_argument(
        '--oracle-text',
        type=str,
        default=None,
        help='Oracle text'
    )
    parser.add_argument(
        '--mana-cost',
        type=str,
        default=None,
        help='Mana cost (e.g., "{R}{G}{W}")'
    )
    parser.add_argument(
        '--type-line',
        type=str,
        default=None,
        help='Type line (e.g., "Creature — Human Wizard")'
    )
    parser.add_argument(
        '--release-year',
        type=str,
        default=None,
        help='Release year'
    )
    parser.add_argument(
        '--rarity',
        type=str,
        default=None,
        help='Rarity (common, uncommon, rare, mythic)'
    )
    parser.add_argument(
        '--set',
        type=str,
        default=None,
        help='Set code'
    )
    parser.add_argument(
        '--power',
        type=str,
        default=None,
        help='Power (for creatures)'
    )
    parser.add_argument(
        '--toughness',
        type=str,
        default=None,
        help='Toughness (for creatures)'
    )
    parser.add_argument(
        '--loyalty',
        type=str,
        default=None,
        help='Loyalty (for planeswalkers)'
    )
    
    # Generation arguments
    parser.add_argument(
        '--num-tokens',
        type=int,
        default=200,
        help='Maximum number of tokens to generate (default: 200)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed. For reproducibility, set a specific seed.'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu). Default: auto-detect'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
    else:
        args.seed = random.randint(0, 1000000)
        print(f"Using random seed: {args.seed}")
    
    # Set device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")
    
    try:
        # Load model and token map
        model, token_map, decoder, block_size, vocab_size = load_model_and_token_map(
            args.model, args.token_map, device
        )
        
        # Create card from arguments
        print("\nCreating card from provided fields...")
        card = create_card_from_args(args)
        
        # Show what fields are provided
        provided_fields = []
        if card.name is not None:
            provided_fields.append("name")
        if card.oracle_text is not None:
            provided_fields.append("oracle_text")
        if card.mana_cost is not None:
            provided_fields.append("mana_cost")
        if card.type_line is not None:
            provided_fields.append("type_line")
        if card.release_year is not None:
            provided_fields.append("release_year")
        if card.rarity is not None:
            provided_fields.append("rarity")
        if card.set_code is not None:
            provided_fields.append("set")
        if card.power is not None:
            provided_fields.append("power")
        if card.toughness is not None:
            provided_fields.append("toughness")
        if card.loyalty is not None:
            provided_fields.append("loyalty")
        
        if provided_fields:
            print(f"Provided fields: {', '.join(provided_fields)}")
        else:
            print("No fields provided - generating complete card from scratch")
        
        # Build FIM prompt: missing fields become sentinels; prompt ends with </FIM><sentinel_0>
        print("\nBuilding FIM context...")
        context_tokens, runs = build_fim_prompt_for_inference(card)
        print(f"Context tokens: {len(context_tokens)} tokens")
        if runs:
            missing_count = sum(len(r) for r in runs)
            print(f"Missing fields (to generate): {missing_count} in {len(runs)} gap(s)")
        
        # If no missing fields, card is complete
        if not runs:
            print("No missing fields - card is complete.")
            print_card(card)
            return
        
        # Convert tokens to IDs for context
        context_token_ids = []
        for token in context_tokens:
            token_id = token_map.get(token, None)
            if token_id is not None:
                context_token_ids.append(token_id)
            else:
                print(f"Warning: Token '{token}' not found in token map, skipping.")
        
        if not context_token_ids:
            print("Warning: No valid tokens in context. Starting with begin_card_token.")
            context_token_ids = [token_map.get(begin_card_token, 0)]
        
        context = torch.tensor([context_token_ids], dtype=torch.long, device=device)
        print(f"Context length: {context.shape[1]} tokens")
        
        # Generate tokens (model continues from <sentinel_0> with gap contents, then </card>)
        print(f"\nGenerating up to {args.num_tokens} tokens...")
        full_sequence, new_tokens = generate_tokens(
            model, context, args.num_tokens, block_size, device
        )
        
        # Decode generated part only (tokens after the prompt)
        all_token_strings = decode_tokens(full_sequence[0], decoder)
        generated_token_strings = all_token_strings[len(context_tokens):]
        print(f"{all_token_strings}")
        
        # Parse generated tail: split by sentinels, parse each chunk into run fields, merge into card
        print("\nParsing generated tokens into card fields...")
        parse_generated_sentinel_tail(generated_token_strings, runs, card)
        
        # Print completed card
        print_card(card)
    
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
