from re import sub
from string import ascii_lowercase, digits

punctuation_characters = '.,\n'
oracle_text_character_whitelist = set(ascii_lowercase + digits + ' "\'\{\}' + punctuation_characters)


class UnsupportedCharacterError(Exception):
    def __init__(self, char):
        self.char = char
        super().__init__(f"Unsupported character: {char}")

def preprocess_oracle_text(oracle_text):
    # Lowercase the text
    oracle_text = oracle_text.lower()

    # remove reminder text (text in between parentheses)
    oracle_text = sub(fr'\(.*\)', '', oracle_text)

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