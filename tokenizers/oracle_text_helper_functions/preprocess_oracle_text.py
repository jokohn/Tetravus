from re import sub
from string import ascii_lowercase, digits

from special_tokens import oracle_text_this_card_token, oracle_text_open_quote_token, oracle_text_close_quote_token

punctuation_characters = '.,\n:'
oracle_text_character_whitelist = set(ascii_lowercase + digits + ' "\'{}<>_' + punctuation_characters)


class UnsupportedCharacterError(Exception):
    def __init__(self, char):
        self.char = char
        super().__init__(f"Unsupported character: {char}")

def preprocess_oracle_text(oracle_text, card_name=None, type_line=None):
    if card_name is not None:
        oracle_text = oracle_text.replace(card_name, oracle_text_this_card_token)
        
        if type_line is not None and 'Legendary' in type_line:
            try:
                first_name = card_name.split(',')[0]
                oracle_text = oracle_text.replace(first_name, oracle_text_this_card_token)
            except IndexError:
                pass

            try:
                first_name = card_name.split(' of')[0]
                oracle_text = oracle_text.replace(first_name, oracle_text_this_card_token)
            except IndexError:
                pass

    # Lowercase the text
    oracle_text = oracle_text.lower()

    # remove reminder text (text in between parentheses)
    oracle_text = sub(fr'\(.*\)', '', oracle_text)
    
    oracle_text = sub(r'\"(.)', oracle_text_open_quote_token + r' \1', oracle_text)
    oracle_text = sub(r'(.)"', r'\1 ' + oracle_text_close_quote_token, oracle_text)
    

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