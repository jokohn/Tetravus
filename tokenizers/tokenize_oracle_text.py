from re import sub

from .oracle_text_helper_functions.preprocess_oracle_text import preprocess_oracle_text
from special_tokens import begin_oracle_text_token, end_oracle_text_token

def tokenize_oracle_text(oracle_text):
    preprocessed_oracle_text = preprocess_oracle_text(oracle_text)
    tokens = [begin_oracle_text_token]
    for word in preprocessed_oracle_text.split(' '):
        if word == '':
            continue
        tokens.append(f'<oracle_text_{word}>')
    tokens.append(end_oracle_text_token)
    return tokens

def detokenize_oracle_text(token_stream):
    start_token = token_stream.peek()
    if start_token != begin_oracle_text_token:
        raise ValueError(f"Expected begin_oracle_text_token, got {start_token}")
    token_stream.advance()
    oracle_text = []
    current_token = token_stream.consume_token()
    while current_token != end_oracle_text_token:
        oracle_text.append(current_token.replace('<oracle_text_', '').replace('>', ''))
        current_token = token_stream.consume_token()
    oracle_text = ' '.join(oracle_text)

    # Remove space between words and following punctuation characters
    oracle_text = sub(r'([a-zA-Z]) ([.,])', r'\1\2', oracle_text)

    # remove spaces around newlines
    oracle_text = sub(r' +\n +', '\n', oracle_text)

    # capitalize the first letter after a newline or period or start of the string
    oracle_text = sub(
        r'(\n|\.|^)( ?)([a-z])',
        lambda match: f'{match.group(1)}{match.group(2)}{match.group(3).upper()}',
        oracle_text)

    return oracle_text