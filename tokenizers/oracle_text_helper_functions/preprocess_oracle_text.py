import json
import os


from re import sub
from string import ascii_lowercase, digits

from special_tokens import oracle_text_this_card_token, oracle_text_open_quote_token, oracle_text_close_quote_token, \
    oracle_text_other_card_token

punctuation_characters = '.,\n:—'
oracle_text_character_whitelist = set(ascii_lowercase + digits + ' "•\'{}<>_+−!-/&' + punctuation_characters)

# Determine the path to the whitelist JSON relative to this file
_whitelist_path = os.path.join(os.path.dirname(__file__), "em_dash_keyword_whitelist.json")

with open(_whitelist_path, "r") as f:
    em_dash_keyword_whitelist = json.load(f)


class UnsupportedCharacterError(Exception):
    def __init__(self, char):
        self.char = char
        super().__init__(f"Unsupported character: {char}")

def preprocess_oracle_text(oracle_text, card_name=None, type_line=None, related_card_names=None):
    if card_name is not None:
        oracle_text = oracle_text.replace(card_name, oracle_text_this_card_token)

        if related_card_names is not None:
            for related_card_name in related_card_names:
                oracle_text = oracle_text.replace(related_card_name, oracle_text_other_card_token)
        
        if type_line is not None and 'Legendary' in type_line:
            for split_term in [',', ' of' , ' the', ' ']:
                try:
                    first_name = card_name.split(split_term)[0]
                    if first_name:
                        oracle_text = oracle_text.replace(first_name, oracle_text_this_card_token)
                except IndexError:
                    pass

    # Lowercase the text
    oracle_text = oracle_text.lower()

    # Split the oracle text into lines
    lines = oracle_text.split('\n')
    processed_lines = []

    for line in lines:
        # Only process for em dash if present in the line
        if '—' in line:
            parts = line.split('—', 1)
            # Check if any whitelist keyword is present in the part before the em dash
            keep_pre_em_dash = False
            for keyword in em_dash_keyword_whitelist:
                if keyword.lower() in parts[0]:
                    keep_pre_em_dash = True
                    break
            if keep_pre_em_dash:
                # Keep the full line, including text before and after
                processed_lines.append(line)
            else:
                pre_em_dash_text = ''
                if '•' in parts[0]:
                    pre_em_dash_text = '• '

                # Remove only the pre-em dash portion, keep what's after the em dash
                post_em_dash = parts[1].lstrip()  # remove leading whitespace after em dash
                # Add em dash back if post text isn't empty
                if post_em_dash:
                    processed_lines.append(pre_em_dash_text + post_em_dash)
        else:
            # No em dash, leave as is
            processed_lines.append(line)

    oracle_text = '\n'.join(processed_lines)

    # remove reminder text (text in between parentheses)
    oracle_text = sub(fr'\(.*\)', '', oracle_text)

    # Semicolons are used to split keyword abilities where the final ability has reminder text.
    # Since we don't support keyword abilities, we need to replace them with commas.
    oracle_text = sub(';', ',', oracle_text)
    
    oracle_text = sub(r'\"([^ ])', oracle_text_open_quote_token + r' \1', oracle_text)
    oracle_text = sub(r'([^ ])\"', r'\1 ' + oracle_text_close_quote_token, oracle_text)
    

    # add spacing before and after punctuation characters
    for punctuation_character in punctuation_characters:
        oracle_text = oracle_text.replace(punctuation_character, f' {punctuation_character} ')

    # strip excess whitespace
    oracle_text = sub(r' +', ' ', oracle_text)
    oracle_text = sub(r'^ ', '', oracle_text)
    oracle_text = sub(r' $', '', oracle_text)

    # Check for unsupported characters
    for char in oracle_text:
        if char not in oracle_text_character_whitelist:
            raise UnsupportedCharacterError(char)

    return oracle_text