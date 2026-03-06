import argparse
import ijson
import itertools
import json
import random
import math

from card import Card
from tokenizers.oracle_text_helper_functions.preprocess_oracle_text import UnsupportedCharacterError
from special_tokens import (
    begin_card_token,
    end_card_token,
    fim_begin_token,
    fim_end_token,
    sentinel_tokens,
)
from fim_utils import (
    get_canonical_fields_for_card,
    build_fim_block,
    CANONICAL_FIELD_ORDER,
)


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


# Weights for sampling k (number of fields to mask): favor 5-6, occasionally 2-4 or 7-9.
# Index i corresponds to k = 2 + i (so k=2..9 for 8 weights).
K_SAMPLE_WEIGHTS = [1, 2, 4, 6, 6, 4, 2, 1]


def sample_k_for_fim(num_fields):
    """
    Sample k (number of fields to mask) with 1 < k < num_fields.
    Distribution favors 5 and 6 when num_fields is large enough.
    """
    if num_fields <= 2:
        return None  # cannot satisfy 1 < k < n
    valid_k = list(range(2, num_fields))
    # Use first (num_fields - 2) weights
    w = K_SAMPLE_WEIGHTS[: len(valid_k)]
    return random.choices(valid_k, weights=w, k=1)[0]


def tokenize_card_file_fim(
    cleaned_cards_file_name,
    seed=None,
    train_test_split=0.8,
    blocks_per_card=10,
):
    """
    Tokenize cards: each card generates blocks_per_card blocks for its target set (train or test).
    Half of the blocks are standard (canonical order), half are FIM. FIM blocks use different
    random masks each time.

    Returns:
        token_map, train_token_blocks, test_token_blocks, metadata
    """
    if seed is not None:
        random.seed(seed)

    token_map = {
        begin_card_token: 0,
        end_card_token: 1,
        fim_begin_token: 2,
        fim_end_token: 3,
    }
    for i, st in enumerate(sentinel_tokens):
        token_map[st] = 4 + i
    token_counter = 4 + len(sentinel_tokens)

    train_token_blocks = []
    test_token_blocks = []

    processed_cards = 0
    failed_cards = 0
    train_cards = 0
    test_cards = 0

    num_standard = blocks_per_card // 2
    num_fim = blocks_per_card - num_standard

    with open(cleaned_cards_file_name, "r") as f:
        for card_dict in ijson.items(f, "item"):
            try:
                card = Card.from_json(None, card_dict)
                canonical_fields = get_canonical_fields_for_card(card)
                if len(canonical_fields) < 3:
                    failed_cards += 1
                    continue

                is_training = random.random() < train_test_split
                target_blocks = train_token_blocks if is_training else test_token_blocks

                standard_block = [begin_card_token] + card.generate_tokens(canonical_fields) + [end_card_token]

                for _ in range(num_standard):
                    for token in standard_block:
                        if token not in token_map:
                            token_map[token] = token_counter
                            token_counter += 1
                    target_blocks.append(standard_block)

                for _ in range(num_fim):
                    k = sample_k_for_fim(len(canonical_fields))
                    if k is None:
                        block_tokens = standard_block
                    else:
                        mask_set = set(random.sample(canonical_fields, k))
                        runs = _compute_runs_for_fim(canonical_fields, mask_set)
                        if len(runs) > len(sentinel_tokens):
                            block_tokens = standard_block
                        else:
                            try:
                                block_tokens = build_fim_block(card, mask_set)
                            except ValueError:
                                block_tokens = standard_block
                    for token in block_tokens:
                        if token not in token_map:
                            token_map[token] = token_counter
                            token_counter += 1
                    target_blocks.append(block_tokens)

                processed_cards += 1
                if is_training:
                    train_cards += 1
                else:
                    test_cards += 1

                if processed_cards % 500 == 0:
                    print(
                        f"Processed card {processed_cards} (train: {train_cards}, test: {test_cards}, "
                        f"blocks: train={len(train_token_blocks)}, test={len(test_token_blocks)})"
                    )

            except (ValueError, UnsupportedCharacterError):
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
        "total_unique_tokens": len(token_map),
    }

    return token_map, train_token_blocks, test_token_blocks, metadata


def _compute_runs_for_fim(canonical_fields, mask_set):
    """Compute runs of consecutive masked fields (same logic as fim_utils._compute_runs)."""
    runs = []
    current_run = []
    for f in canonical_fields:
        if f in mask_set:
            current_run.append(f)
        else:
            if current_run:
                runs.append(current_run)
                current_run = []
    if current_run:
        runs.append(current_run)
    return runs


def generate_field_permutations(fields, num_permutations):
    """
    Generate N permutations of field orderings efficiently.
    """
    total_permutations = math.factorial(len(fields))
    
    if num_permutations > total_permutations:
        # If we need more than possible, generate all once and sample
        all_permutations = list(itertools.permutations(fields))
        return [list(random.choice(all_permutations)) for _ in range(num_permutations)]
    else:
        # For small num_permutations, generate random permutations directly
        if num_permutations < total_permutations // 2:
            # More efficient: generate random permutations directly
            seen = set()
            permutations = []
            fields_list = list(fields)
            while len(permutations) < num_permutations:
                shuffled = fields_list.copy()
                random.shuffle(shuffled)
                perm_tuple = tuple(shuffled)
                if perm_tuple not in seen:
                    seen.add(perm_tuple)
                    permutations.append(list(shuffled))
            return permutations
        else:
            # If we need many permutations, materialize and sample
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
    
    token_map = {
        begin_card_token: 0,
        end_card_token: 1,
    }
    train_token_blocks = []
    test_token_blocks = []
    token_counter = 2
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
                        token_block = [begin_card_token]
                        for token in tokens:
                            if token not in token_map:
                                token_map[token] = token_counter
                                token_counter += 1
                            token_block.append(token)
                        token_block.append(end_card_token)
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
    Randomly shuffle token blocks and encode as chr-encoded strings.

    Args:
        token_blocks: List of token blocks (each is a list of token strings)
        token_map: Dict mapping token strings to numeric IDs

    Returns:
        List of chr-encoded strings (one per token block).
    """
    shuffled_blocks = token_blocks.copy()
    random.shuffle(shuffled_blocks)
    encoded_blocks = []
    for block in shuffled_blocks:
        token_ids = [token_map[token] for token in block]
        char_string = "".join(chr(token_id) for token_id in token_ids)
        encoded_blocks.append(char_string)
    return encoded_blocks


def write_tokenized_output(
    train_blocks_encoded,
    test_blocks_encoded,
    token_map,
    train_output_text_file,
    test_output_text_file,
    output_map_file,
    metadata,
):
    """Write chr-encoded token blocks to text files and token map to JSON file."""
    with open(train_output_text_file, "w", encoding="utf-8") as f:
        for encoded_block in train_blocks_encoded:
            f.write(encoded_block)

    with open(test_output_text_file, "w", encoding="utf-8") as f:
        for encoded_block in test_blocks_encoded:
            f.write(encoded_block)

    output_data = {"token_map": token_map, "metadata": metadata}
    with open(output_map_file, "w") as f:
        json.dump(output_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize a cleaned cards JSON file with FIM (50%% Standard / 50%% FIM) for LLM training."
    )
    parser.add_argument(
        "cleaned_cards_file_name",
        help="Path to the input cleaned cards JSON file",
    )
    parser.add_argument(
        "train_output_text_file",
        help="Path to the output training text file (chr-encoded token blocks, UTF-8)",
    )
    parser.add_argument(
        "test_output_text_file",
        help="Path to the output test text file (chr-encoded token blocks, UTF-8)",
    )
    parser.add_argument(
        "output_map_file",
        help="Path to the output token map JSON file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--train-test-split",
        type=float,
        default=0.8,
        help="Fraction of cards to assign to training set (default: 0.8)",
    )
    parser.add_argument(
        "--blocks-per-card",
        type=int,
        default=10,
        help="Number of blocks to generate per card (half standard, half FIM) (default: 10)",
    )
    args = parser.parse_args()

    if not (0.0 < args.train_test_split < 1.0):
        parser.error("--train-test-split must be between 0.0 and 1.0")
    if args.blocks_per_card < 1:
        parser.error("--blocks-per-card must be at least 1")

    print("Tokenizing cards with FIM (50% Standard / 50% FIM)...")
    print(f"Blocks per card: {args.blocks_per_card} (standard: {args.blocks_per_card // 2}, FIM: {args.blocks_per_card - args.blocks_per_card // 2})")
    print(f"Train/test split: {args.train_test_split:.1%} training, {1 - args.train_test_split:.1%} test")
    print(f"Input file: {args.cleaned_cards_file_name}")

    token_map, train_token_blocks, test_token_blocks, metadata = tokenize_card_file_fim(
        args.cleaned_cards_file_name,
        seed=args.seed,
        train_test_split=args.train_test_split,
        blocks_per_card=args.blocks_per_card,
    )

    print(f"\nTokenization complete:")
    print(f"  Processed cards: {metadata['processed_cards']}")
    print(f"  Failed cards: {metadata['failed_cards']}")
    if metadata["processed_cards"] > 0:
        print(
            f"  Training cards: {metadata['train_cards']} "
            f"({metadata['train_cards']/metadata['processed_cards']*100:.1f}%)"
        )
        print(
            f"  Test cards: {metadata['test_cards']} "
            f"({metadata['test_cards']/metadata['processed_cards']*100:.1f}%)"
        )
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

    print("Writing output files...")
    write_tokenized_output(
        train_encoded_blocks,
        test_encoded_blocks,
        token_map,
        args.train_output_text_file,
        args.test_output_text_file,
        args.output_map_file,
        metadata,
    )

    print("\nOutput saved:")
    print(f"  Training token blocks: {args.train_output_text_file}")
    print(f"  Test token blocks: {args.test_output_text_file}")
    print(f"  Token map: {args.output_map_file}")


if __name__ == "__main__":
    main()

