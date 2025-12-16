import argparse
import ijson
import json

from card import Card
from tokenizers.oracle_text_helper_functions.preprocess_oracle_text import UnsupportedCharacterError

def tokenize_card_file(cleaned_cards_file_name):
    token_map = {}
    token_metadata = {}
    token_string = ''
    token_counter = 0
    processed_cards, failed_cards = 0, 0
    token_count = 0
    with open(cleaned_cards_file_name, "r") as f:
        for card_dict in ijson.items(f, 'item'):
            try:
                card = Card.from_json(None, card_dict)
                tokens = card.generate_tokens(["name", "oracle_text", "mana_cost", "type_line", "release_year", "rarity", "set", "power", "toughness"])
            except (ValueError, UnsupportedCharacterError) as e:
                failed_cards += 1
                continue
            processed_cards += 1
            if processed_cards % 500 == 0:
                print(f"Processed card {processed_cards} of {processed_cards + failed_cards}")
            for token in tokens:
                token_count += 1
                token_metadata[token] = token_metadata.setdefault(token, 0) + 1
                if token not in token_map:
                    token_map[token] = token_counter
                    token_counter += 1
    return token_map, token_string, token_metadata, processed_cards, failed_cards, token_count

def tokenize_card_file_and_save(cleaned_cards_file_name, output_file_name):
    token_map, token_string, token_metadata, processed_cards, failed_cards, token_count = tokenize_card_file(cleaned_cards_file_name)
    with open(output_file_name, "w") as f:
        json.dump({
            "token_map": token_map,
            "token_string": token_string,
            "token_metadata": token_metadata,
            "processed_cards": processed_cards,
            "failed_cards": failed_cards,
            "total_tokens": max(token_map.values()),
            "tokens_generated": token_count,  
        }, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize a cleaned cards JSON file and save the token map.")
    parser.add_argument("cleaned_cards_file_name", help="Path to the input cleaned cards JSON file")
    parser.add_argument("output_file_name", help="Path to the output token map JSON file")
    
    args = parser.parse_args()
    
    tokenize_card_file_and_save(args.cleaned_cards_file_name, args.output_file_name)
    print(f"Tokenization complete. Output saved to {args.output_file_name}")
