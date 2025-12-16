from re import sub

from .oracle_text_helper_functions.preprocess_oracle_text import preprocess_oracle_text
from .tokenize_mana_cost import tokenize_mana_cost, detokenize_mana_cost
from special_tokens import begin_oracle_text_token, end_oracle_text_token, begin_oracle_text_mana_cost_token, \
    end_oracle_text_mana_cost_token, oracle_text_this_card_token

def tokenize_oracle_text(oracle_text, card_name=None, type_line=None):
    preprocessed_oracle_text = preprocess_oracle_text(oracle_text, card_name, type_line)
    tokens = [begin_oracle_text_token]
    for word in preprocessed_oracle_text.split(' '):
        if word == '':
            continue
        elif word.startswith('{') and word.endswith('}'):
            tokens.extend(tokenize_mana_cost(word, is_orcale_text_mana_cost=True))
        elif word.startswith('<') and word.endswith('>'):
            tokens.append(word)
        else:
            tokens.append(f'<oracle_text_{word}>')
    tokens.append(end_oracle_text_token)
    return tokens

def detokenize_oracle_text(token_stream, card_name=None):
    start_token = token_stream.peek()
    if start_token != begin_oracle_text_token:
        raise ValueError(f"Expected begin_oracle_text_token, got {start_token}")
    token_stream.advance()
    oracle_text = []
    current_token = token_stream.consume_token()
    while current_token != end_oracle_text_token:
        if current_token == begin_oracle_text_mana_cost_token:
            token_stream.jump_by(-1)
            oracle_text.append(detokenize_mana_cost(token_stream, is_orcale_text_mana_cost=True))
        elif current_token == oracle_text_this_card_token:
            oracle_text.append(card_name)
        else:
            oracle_text.append(current_token.replace('<oracle_text_', '').replace('>', ''))
        current_token = token_stream.consume_token()
    oracle_text = ' '.join(oracle_text)

    # Remove space between words and following punctuation characters
    oracle_text = sub(r'([a-zA-Z\}]) ([.,:])', r'\1\2', oracle_text)


    # remove spaces around newlines
    oracle_text = sub(r' +\n +', '\n', oracle_text)

    # capitalize the first letter after a newline or period or start of the string
    oracle_text = sub(
        r'(\n|\.|^|:)( ?)([a-z])',
        lambda match: f'{match.group(1)}{match.group(2)}{match.group(3).upper()}',
        oracle_text)
    return oracle_text