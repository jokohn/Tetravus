from special_tokens import begin_name_token, end_name_token

def tokenize_name(name):
    
    tokens = []
    for i in name:
        tokens.append(f'<name_char_{i}>')
    return [begin_name_token] + tokens + [end_name_token]

def detokenize_name(token_stream):
    name = []
    start_token = token_stream.peek()
    if start_token != begin_name_token:
        raise ValueError(f"Expected begin_name_token, got {start_token}")
    token_stream.advance()
    current_token = token_stream.consume_token()
    while current_token != end_name_token:
        if (current_token.startswith('<name_char_') and current_token.endswith('>')):
            name.append(current_token.replace('<name_char_', '').replace('>', ''))
            current_token = token_stream.consume_token()
        elif current_token == begin_name_token:
            # The model seems to regularly insert name tokens into name blocks, so we skip them
            current_token = token_stream.consume_token()
        else:
            raise ValueError(f"Malformed token stream: unexpected token {current_token!r} in name block")
    return ''.join(name)
