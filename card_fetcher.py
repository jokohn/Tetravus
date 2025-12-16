import ijson
import json
import requests
import time

from card import Card

def fetch_cards_and_save(output_file_name=None):
    bulk_data_response = requests.get("https://api.scryfall.com/bulk-data")
    bulk_data = bulk_data_response.json()
    for item in bulk_data['data']:
        if item["type"] == "all_cards":
            oracle_cards_url = item["download_uri"]
            current_time = time.strftime("%Y%m%d_%H%M%S")
            if output_file_name is None:
                output_file_name = f"{current_time}_raw_cards_file.json"
            with open(output_file_name, "wb") as f:
                print(f"Downloading cards from {oracle_cards_url}")
                oracle_cards_response = requests.get(oracle_cards_url)
                print(f"Downloaded {oracle_cards_response.status_code} cards")
                print(f"Writing to file {output_file_name}")
                chunk_count = 0
                for chunk in oracle_cards_response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        chunk_count += 1
                        if chunk_count % 10000 == 0:
                            print(f"Written {chunk_count} chunks to file")

def card_cleaner(raw_cards_file_name):
    print(f"Cleaning cards from {raw_cards_file_name}")
    f = open(raw_cards_file_name, "rb")
    cards_by_name = {}
    card_count = 0
    for card_dict in ijson.items(f, 'item'):
        if card_dict['lang'] != 'en':
            continue
        elif card_dict['legalities']['vintage'] == 'not_legal':
            continue
        elif card_dict['legalities']['vintage'] == 'banned':
            continue
        elif card_dict.get('card_faces'):
            continue
        card = Card.from_json(None, card_dict).to_json()
        card['released_at'] = card_dict['released_at']
        card['release_date'] = time.strptime(card_dict['released_at'], '%Y-%m-%d')
        if old_card := cards_by_name.get(card['name']):
            if old_card['release_date'] > card['release_date']:
                cards_by_name[card['name']] = card
        else:
            cards_by_name[card['name']] = card
        card_count += 1
        if card_count % 250 == 0:
            print(f"Processed {card_count} cards")
    f.close()
    return list(cards_by_name.values())

def clean_cards(raw_cards_file_name):
    cleaned_cards = card_cleaner(raw_cards_file_name)
    for card in cleaned_cards:
        del card['release_date']
    with open(f"{raw_cards_file_name.split('_')[0]}_{raw_cards_file_name.split('_')[1]}_cleaned.json", "w") as f:
        json.dump(cleaned_cards, f)