import argparse
import base64
import ijson
import itertools
import json
import random
import struct
import math

from card import Card
from tokenizers.oracle_text_helper_functions.preprocess_oracle_text import UnsupportedCharacterError


def get_available_fields(card):
    """
    Determine which fields are available for a card.
    
    Required fields: name, mana_cost, type_line, release_year, rarity, set
    Optional fields: oracle_text (if not None), power (if not None), 
                     toughness (if not None), loyalty (if not None)
    
    Args:
        card: Card object
        
    Returns:
        List of field names available for this card
    """
    fields = ["name", "mana_cost", "type_line", "release_year", "rarity", "set", "oracle_text"]
    
    if card.power is not None:
        fields.append("power")
    
    if card.toughness is not None:
        fields.append("toughness")
    
    if card.loyalty is not None:
        fields.append("loyalty")
    
    return fields


def generate_field_permutations(fields, num_permutations):
    """
    Generate N permutations of field orderings.
    
    When num_permutations <= total possible permutations, returns N unique permutations.
    When num_permutations > total possible permutations, returns all unique permutations
    plus randomly selected duplicates to reach the requested count.
    
    Args:
        fields: List of field names
        num_permutations: Number of permutations to generate
        
    Returns:
        List of field order lists (each is a permutation of fields).
        Uniqueness is guaranteed only when num_permutations <= factorial(len(fields)).
    """
    # Calculate total number of possible permutations
    total_permutations = math.factorial(len(fields))
    
    # If we need more permutations than possible, generate all and repeat
    if num_permutations > total_permutations:
        all_permutations = list(itertools.permutations(fields))
        # Repeat randomly to reach num_permutations
        permutations = []
        for _ in range(num_permutations):
            permutations.append(list(random.choice(all_permutations)))
        return permutations
    else:
        # Generate N unique permutations
        all_permutations = list(itertools.permutations(fields))
        selected = random.sample(all_permutations, num_permutations)
        return [list(p) for p in selected]


def tokenize_card_file_permuted(cleaned_cards_file_name, num_permutations=10, seed=None, train_test_split=0.8):
    """
    Main processing function that tokenizes cards with field permutations.
    
    Args:
        cleaned_cards_file_name: Path to input cleaned cards JSON file
        num_permutations: Number of field permutations per card
        seed: Random seed for reproducibility
        train_test_split: Fraction of cards to assign to training set (default: 0.8)
        
    Returns:
        Tuple of (token_map, train_token_blocks, test_token_blocks, metadata)
        - token_map: dict mapping token strings to numeric IDs
        - train_token_blocks: list of token blocks for training (each is a list of token strings)
        - test_token_blocks: list of token blocks for test (each is a list of token strings)
        - metadata: dict with processed_cards, failed_cards, train_blocks, test_blocks, train_cards, test_cards
    """
    if seed is not None:
        random.seed(seed)
    
    token_map = {}
    train_token_blocks = []
    test_token_blocks = []
    token_counter = 0
    processed_cards = 0
    failed_cards = 0
    train_cards = 0
    test_cards = 0
    
    with open(cleaned_cards_file_name, "r") as f:
        for card_dict in ijson.items(f, 'item'):
            try:
                card = Card.from_json(None, card_dict)
                available_fields = get_available_fields(card)
                
                # Randomly assign card to training or test set
                is_training = random.random() < train_test_split
                target_blocks = train_token_blocks if is_training else test_token_blocks
                
                # Generate field permutations for this card
                field_permutations = generate_field_permutations(available_fields, num_permutations)
                
                # Tokenize each permutation
                for field_order in field_permutations:
                    try:
                        tokens = card.generate_tokens(field_order)
                        
                        # Build token_map and convert tokens to IDs
                        token_block = []
                        for token in tokens:
                            if token not in token_map:
                                token_map[token] = token_counter
                                token_counter += 1
                            token_block.append(token)
                        
                        target_blocks.append(token_block)
                    except (ValueError, UnsupportedCharacterError) as e:
                        # Skip this permutation if it fails
                        continue
                
                processed_cards += 1
                if is_training:
                    train_cards += 1
                else:
                    test_cards += 1
                    
                if processed_cards % 500 == 0:
                    print(f"Processed card {processed_cards} (train: {train_cards}, test: {test_cards}, blocks: train={len(train_token_blocks)}, test={len(test_token_blocks)})")
                    
            except (ValueError, UnsupportedCharacterError) as e:
                failed_cards += 1
                continue
    
    metadata = {
        "processed_cards": processed_cards,
        "failed_cards": failed_cards,
        "train_cards": train_cards,
        "test_cards": test_cards,
        "train_blocks": len(train_token_blocks),
        "test_blocks": len(test_token_blocks),
        "total_blocks": len(train_token_blocks) + len(test_token_blocks),
        "total_unique_tokens": len(token_map)
    }
    
    return token_map, train_token_blocks, test_token_blocks, metadata


def shuffle_and_encode_token_blocks(token_blocks, token_map):
    """
    Randomly shuffle token blocks and encode as base64.
    
    Args:
        token_blocks: List of token blocks (each is a list of token strings)
        token_map: Dict mapping token strings to numeric IDs
        
    Returns:
        List of base64-encoded strings (one per token block)
    """
    # Shuffle the blocks randomly
    shuffled_blocks = token_blocks.copy()
    random.shuffle(shuffled_blocks)
    
    encoded_blocks = []
    
    for block in shuffled_blocks:
        # Convert token strings to numeric IDs
        token_ids = [token_map[token] for token in block]
        
        # Pack token IDs as bytes (using 4-byte integers)
        # This supports up to 2^32 unique tokens
        packed_bytes = b''.join(struct.pack('>I', token_id) for token_id in token_ids)
        
        # Encode as base64
        encoded = base64.b64encode(packed_bytes).decode('ascii')
        encoded_blocks.append(encoded)
    
    return encoded_blocks


def write_tokenized_output(train_blocks_encoded, test_blocks_encoded, token_map, 
                          train_output_text_file, test_output_text_file, output_map_file, metadata):
    """
    Write base64-encoded token blocks to text files and token map to JSON file.
    
    Args:
        train_blocks_encoded: List of base64-encoded token block strings for training
        test_blocks_encoded: List of base64-encoded token block strings for test
        token_map: Dict mapping token strings to numeric IDs
        train_output_text_file: Path for output training text file (one block per line)
        test_output_text_file: Path for output test text file (one block per line)
        output_map_file: Path for output token map JSON file
        metadata: Dict with processing metadata
    """
    # Write training token blocks to text file (one per line)
    with open(train_output_text_file, 'w') as f:
        for encoded_block in train_blocks_encoded:
            f.write(encoded_block + '\n')
    
    # Write test token blocks to text file (one per line)
    with open(test_output_text_file, 'w') as f:
        for encoded_block in test_blocks_encoded:
            f.write(encoded_block + '\n')
    
    # Write token map and metadata to JSON file
    output_data = {
        "token_map": token_map,
        "metadata": metadata
    }
    
    with open(output_map_file, 'w') as f:
        json.dump(output_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize a cleaned cards JSON file with field permutations for LLM training."
    )
    parser.add_argument(
        "cleaned_cards_file_name",
        help="Path to the input cleaned cards JSON file"
    )
    parser.add_argument(
        "train_output_text_file",
        help="Path to the output training text file (base64-encoded token blocks, one per line)"
    )
    parser.add_argument(
        "test_output_text_file",
        help="Path to the output test text file (base64-encoded token blocks, one per line)"
    )
    parser.add_argument(
        "output_map_file",
        help="Path to the output token map JSON file"
    )
    parser.add_argument(
        "--num-permutations",
        type=int,
        default=10,
        help="Number of field permutations per card (default: 10)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--train-test-split",
        type=float,
        default=0.8,
        help="Fraction of cards to assign to training set (default: 0.8)"
    )
    
    args = parser.parse_args()
    
    if not (0.0 < args.train_test_split < 1.0):
        parser.error("--train-test-split must be between 0.0 and 1.0")
    
    print(f"Tokenizing cards with {args.num_permutations} permutations per card...")
    print(f"Train/test split: {args.train_test_split:.1%} training, {1 - args.train_test_split:.1%} test")
    print(f"Input file: {args.cleaned_cards_file_name}")
    
    token_map, train_token_blocks, test_token_blocks, metadata = tokenize_card_file_permuted(
        args.cleaned_cards_file_name,
        args.num_permutations,
        args.seed,
        args.train_test_split
    )
    
    print(f"\nTokenization complete:")
    print(f"  Processed cards: {metadata['processed_cards']}")
    print(f"  Failed cards: {metadata['failed_cards']}")
    if metadata['processed_cards'] > 0:
        print(f"  Training cards: {metadata['train_cards']} ({metadata['train_cards']/metadata['processed_cards']*100:.1f}%)")
        print(f"  Test cards: {metadata['test_cards']} ({metadata['test_cards']/metadata['processed_cards']*100:.1f}%)")
    else:
        print(f"  Training cards: {metadata['train_cards']}")
        print(f"  Test cards: {metadata['test_cards']}")
    print(f"  Training blocks: {metadata['train_blocks']}")
    print(f"  Test blocks: {metadata['test_blocks']}")
    print(f"  Total blocks: {metadata['total_blocks']}")
    print(f"  Unique tokens: {metadata['total_unique_tokens']}")
    
    print("\nShuffling and encoding token blocks...")
    train_encoded_blocks = shuffle_and_encode_token_blocks(train_token_blocks, token_map)
    test_encoded_blocks = shuffle_and_encode_token_blocks(test_token_blocks, token_map)
    
    print(f"Writing output files...")
    write_tokenized_output(
        train_encoded_blocks,
        test_encoded_blocks,
        token_map,
        args.train_output_text_file,
        args.test_output_text_file,
        args.output_map_file,
        metadata
    )
    
    print(f"\nOutput saved:")
    print(f"  Training token blocks: {args.train_output_text_file}")
    print(f"  Test token blocks: {args.test_output_text_file}")
    print(f"  Token map: {args.output_map_file}")


if __name__ == "__main__":
    main()

