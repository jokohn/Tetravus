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
from fim_utils import (
    build_inference_prompt_for_leftmost_gap,
    find_leftmost_inference_gap,
    get_canonical_fields_for_card,
    normalize_inference_string_field,
    open_field_prefix_tokens,
    FIELD_END_TAG,
)

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
    with open(token_map_path, 'r') as f:
        token_map_data = json.load(f)
        token_map = token_map_data['token_map']
        decoder = {v: k for k, v in token_map.items()}
        vocab_size = len(token_map)
    
    # Load model
    model = torch.load(model_path, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()  # Set to evaluation mode

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    
    # Extract block_size from model (RoPE models use model.block_size; legacy checkpoints use position_embedding_table)
    if hasattr(model, 'block_size'):
        block_size = model.block_size
    else:
        block_size = model.position_embedding_table.weight.shape[0]
    
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


def generate_tokens(model, context, num_tokens, block_size, temperature, top_k, device):
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
        generated = model.generate(context, max_new_tokens=num_tokens, block_size=block_size, temperature=temperature, top_k=top_k)
    
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
    card, _ = create_inference_card_from_args(args)
    return card


def _inference_warn(msg):
    print(msg, file=sys.stderr)


def create_inference_card_from_args(args):
    """
    Build Card from CLI args and track fields that end with the inference '...' convention.

    Returns:
        (card, partial_fields) where partial_fields is a set of canonical field names
        still needing in-block completion.
    """
    partial_fields = set()

    def norm(field_name, raw):
        if raw is None:
            return None
        stored, is_partial = normalize_inference_string_field(
            field_name, raw, warn_fn=_inference_warn
        )
        if is_partial:
            partial_fields.add(field_name)
        return stored

    card = Card(
        name=norm("name", args.name),
        oracle_text=norm("oracle_text", args.oracle_text),
        mana_cost=norm("mana_cost", args.mana_cost),
        type_line=norm("type_line", args.type_line),
        release_year=norm("release_year", args.release_year),
        rarity=norm("rarity", args.rarity),
        set_name=norm("set", args.set),
        power=norm("power", args.power),
        toughness=norm("toughness", args.toughness),
        loyalty=norm("loyalty", args.loyalty),
    )
    return card, partial_fields


def inference_is_complete(card, partial_fields):
    """True when required fields are filled and no ellipsis-partial fields remain."""
    ok, _ = card.is_complete()
    return ok and len(partial_fields) == 0


def merge_open_field_completion(card, open_field, prefix_token_list, generated_token_strings):
    """
    Parse tokens for one open field: prefix (already in context) + generated suffix through end tag.
    Updates card in place; removes open_field from completion needs on success.

    Returns:
        True if end tag was found and the field parsed; False otherwise.
    """
    end_tag = FIELD_END_TAG[open_field]
    end_i = None
    for i, t in enumerate(generated_token_strings):
        if t == end_tag:
            end_i = i
            break
    if end_i is None:
        return False
    chunk = prefix_token_list + generated_token_strings[: end_i + 1]
    stream = TokenStream(chunk)
    try:
        if open_field == "name":
            card.name = detokenize_name(stream)
        elif open_field == "mana_cost":
            card.mana_cost = detokenize_mana_cost(stream)
        elif open_field == "type_line":
            card.type_line = detokenize_type_line(stream)
        elif open_field == "oracle_text":
            card.oracle_text = detokenize_oracle_text(stream, card_name=card.name)
        else:
            return False
    except (ValueError, IndexError):
        return False
    return True


def _token_matches_expected_field_start(token, field_name):
    if field_name == "name":
        return token == begin_name_token
    if field_name == "mana_cost":
        return token == begin_mana_cost_token
    if field_name == "type_line":
        return token == begin_type_line_token
    if field_name == "oracle_text":
        return token == begin_oracle_text_token
    if field_name == "release_year":
        return token.startswith("<release_year_")
    if field_name == "rarity":
        return token.startswith("<rarity_")
    if field_name == "set":
        return token.startswith("<set_")
    if field_name == "power":
        return token.startswith("<power_")
    if field_name == "toughness":
        return token.startswith("<toughness_")
    if field_name == "loyalty":
        return token.startswith("<loyalty_")
    return False


def _skip_extraneous_field_block(stream, card):
    """
    If the stream starts with a full card field block, consume it without updating card.
    Used when the model echoes context fields before the gap content we need to parse.
    Returns True if something was consumed.
    """
    if not stream.has_next():
        return False
    p = stream.peek()
    try:
        if p == begin_name_token:
            detokenize_name(stream)
            return True
        if p == begin_mana_cost_token:
            detokenize_mana_cost(stream)
            return True
        if p == begin_type_line_token:
            detokenize_type_line(stream)
            return True
        if p == begin_oracle_text_token:
            detokenize_oracle_text(stream, card_name=card.name)
            return True
        if p.startswith("<release_year_"):
            detokenize_release_year(stream)
            return True
        if p.startswith("<rarity_"):
            detokenize_rarity(stream)
            return True
        if p.startswith("<set_"):
            detokenize_set_name(stream)
            return True
        if p.startswith("<power_"):
            detokenize_power(stream)
            return True
        if p.startswith("<toughness_"):
            detokenize_toughness(stream)
            return True
        if p.startswith("<loyalty_"):
            detokenize_loyalty(stream)
            return True
        if p in (begin_card_token, end_card_token, fim_begin_token, fim_end_token):
            stream.advance()
            return True
        if p in sentinel_tokens:
            stream.advance()
            return True
    except (ValueError, IndexError):
        return False
    return False


def _sync_stream_to_next_field(stream, field_name, card, max_steps):
    """Advance past echoed field blocks until peek matches the expected field opener or stream ends."""
    steps = 0
    while stream.has_next() and not _token_matches_expected_field_start(stream.peek(), field_name):
        if not _skip_extraneous_field_block(stream, card):
            break
        steps += 1
        if steps > max_steps:
            break


def _parse_chunk_into_fields(chunk_tokens, field_names, card):
    """
    Try to parse a chunk of tokens as a sequence of fields (e.g. one run).
    For each field name in order, attempt to detokenize and set on card.
    The model may prefix the chunk with fields already present in context; those are skipped.
    Stops on first parse failure or when stream is consumed.

    Returns:
        Number of fields successfully parsed.
    """
    if not chunk_tokens or not field_names:
        return 0
    stream = TokenStream(chunk_tokens)
    max_sync_steps = max(len(chunk_tokens), 1)
    parsed = 0
    for field_name in field_names:
        if field_name == "power" and not card.needs_creature_stats():
            if stream.has_next() and stream.peek().startswith("<power_"):
                try:
                    detokenize_power(stream)
                except (ValueError, IndexError):
                    break
            continue
        if field_name == "toughness" and not card.needs_creature_stats():
            if stream.has_next() and stream.peek().startswith("<toughness_"):
                try:
                    detokenize_toughness(stream)
                except (ValueError, IndexError):
                    break
            continue
        if field_name == "loyalty" and not card.needs_planeswalker_loyalty():
            if stream.has_next() and stream.peek().startswith("<loyalty_"):
                try:
                    detokenize_loyalty(stream)
                except (ValueError, IndexError):
                    break
            continue
        if not stream.has_next():
            break
        _sync_stream_to_next_field(stream, field_name, card, max_sync_steps)
        if not stream.has_next() or not _token_matches_expected_field_start(stream.peek(), field_name):
            break
        try:
            if field_name == "name":
                card.name = detokenize_name(stream)
                parsed += 1
            elif field_name == "mana_cost":
                card.mana_cost = detokenize_mana_cost(stream)
                parsed += 1
            elif field_name == "type_line":
                card.type_line = detokenize_type_line(stream)
                parsed += 1
            elif field_name == "oracle_text":
                card.oracle_text = detokenize_oracle_text(stream, card_name=card.name)
                parsed += 1
            elif field_name == "release_year":
                card.release_year = detokenize_release_year(stream)
                parsed += 1
            elif field_name == "rarity":
                card.rarity = detokenize_rarity(stream)
                parsed += 1
            elif field_name == "set":
                card.set_code = detokenize_set_name(stream)
                parsed += 1
            elif field_name == "power":
                card.power = detokenize_power(stream)
                parsed += 1
            elif field_name == "toughness":
                card.toughness = detokenize_toughness(stream)
                parsed += 1
            elif field_name == "loyalty":
                card.loyalty = detokenize_loyalty(stream)
                parsed += 1
            else:
                print(field_name)
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
        runs: List of lists of field names (from build_inference_prompt_for_leftmost_gap / FIM training)
        card: Card to update with parsed fields (modified in place)
    """
    chunks = _split_generated_by_sentinels(generated_tokens)
    #print(f"Chunks: {chunks}")
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
        description=(
            "Load a trained model and generate card fields from partial card information. "
            "For name, mana cost, type line, and oracle text you may end the value with '...' "
            "(or pass exactly '...') to let the model finish that field's token block; invalid "
            "'...' placement is treated as a literal string (see stderr warning)."
        )
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
        help='Card name. Trailing ... or exactly ... completes the name block inside <name>...</name>.',
    )
    parser.add_argument(
        '--oracle-text',
        type=str,
        default=None,
        help='Oracle text. Trailing ... or exactly ... completes the oracle_text block.',
    )
    parser.add_argument(
        '--mana-cost',
        type=str,
        default=None,
        help='Mana cost (e.g., "{R}{G}{W}"). Trailing ... or exactly ... completes the mana_cost block.',
    )
    parser.add_argument(
        '--type-line',
        type=str,
        default=None,
        help='Type line (e.g., "Creature — Human Wizard"). Trailing ... or exactly ... completes the block.',
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
        '--max-retries',
        type=int,
        default=10,
        help='Max number of generate+parse rounds to fill missing fields (default: 10)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu). Default: auto-detect'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='Temperature for generation. Default: 0.8'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=50,
        help='Top-k for generation. Default: 50'
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
        
        # Create card from arguments (including optional '...' in-block completion markers)
        print("\nCreating card from provided fields...")
        card, partial_fields = create_inference_card_from_args(args)
        print(card)
        
        # Show what fields the user passed on the CLI (includes ... markers)
        provided_fields = []
        if args.name is not None:
            provided_fields.append("name")
        if args.oracle_text is not None:
            provided_fields.append("oracle_text")
        if args.mana_cost is not None:
            provided_fields.append("mana_cost")
        if args.type_line is not None:
            provided_fields.append("type_line")
        if args.release_year is not None:
            provided_fields.append("release_year")
        if args.rarity is not None:
            provided_fields.append("rarity")
        if args.set is not None:
            provided_fields.append("set")
        if args.power is not None:
            provided_fields.append("power")
        if args.toughness is not None:
            provided_fields.append("toughness")
        if args.loyalty is not None:
            provided_fields.append("loyalty")
        
        if provided_fields:
            print(f"Provided fields: {', '.join(provided_fields)}")
        else:
            print("No fields provided - generating complete card from scratch")
        if partial_fields:
            print(
                "In-block completion (...): "
                + ", ".join(sorted(partial_fields))
            )
        
        # Iterative generate+parse until card is complete or max_retries reached
        round_num = 0
        while round_num < args.max_retries:
            partial_fields.intersection_update(get_canonical_fields_for_card(card))
            plan = build_inference_prompt_for_leftmost_gap(card, partial_fields)
            if plan.parse_mode == "complete_card":
                print("\nNothing left to infer for the current card.")
                break

            print("\nBuilding inference context...")
            context_tokens = plan.prompt_tokens
            print(f"Context tokens: {len(context_tokens)} tokens")
            
            round_num += 1
            print(f"\nRound {round_num}/{args.max_retries}")
            
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
            
            full_sequence, new_tokens = generate_tokens(
                model, context, args.num_tokens, block_size, args.temperature, args.top_k, device
            )
            
            # Decode generated part only (tokens after the prompt)
            all_token_strings = decode_tokens(full_sequence[0], decoder)
            generated_token_strings = all_token_strings[len(context_tokens):]

            if plan.parse_mode == "fim_sentinel_tail":
                parse_generated_sentinel_tail(generated_token_strings, plan.runs, card)
            elif plan.parse_mode == "continue_open_field":
                prefix_toks = open_field_prefix_tokens(card, plan.open_field)
                merged = merge_open_field_completion(
                    card, plan.open_field, prefix_toks, generated_token_strings
                )
                if merged:
                    partial_fields.discard(plan.open_field)
                else:
                    print(
                        f"Warning: did not find expected closing tag {plan.end_tag!r} "
                        f"for field {plan.open_field!r} in this round's generation.",
                        file=sys.stderr,
                    )

            if inference_is_complete(card, partial_fields):
                break
        
        if (
            not inference_is_complete(card, partial_fields)
            and round_num >= args.max_retries
        ):
            nxt = find_leftmost_inference_gap(card, partial_fields)
            print(
                f"\nStopped after max retries ({args.max_retries}) with inference still incomplete "
                f"(next gap: {nxt!r}; partial ellipsis fields: {sorted(partial_fields)}).",
                file=sys.stderr,
            )
        
        # Print card (complete or partial)
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
