from special_tokens import begin_type_line_token, end_type_line_token, subtype_break_token

def tokenize_type_line(type_line_text):
    tokens = [begin_type_line_token]
    types, subtypes = type_line_text.split(' — ') if ' — ' in type_line_text else (type_line_text, '')
    for type in types.split(' '):
        tokens.append(f'<type_{type}>')
    if subtypes:
        tokens.append(subtype_break_token)
        for subtype in subtypes.split(' '):
            tokens.append(f'<subtype_{subtype}>')
    tokens.append(end_type_line_token)
    return tokens

def detokenize_type_line(token_stream):
    start_token = token_stream.peek()
    if start_token != begin_type_line_token:
        raise ValueError(f"Expected begin_type_line_token, got {start_token}")
    token_stream.advance()
    type_line_text = []
    current_token = token_stream.consume_token()
    while current_token != end_type_line_token:
        if current_token == subtype_break_token:
            type_line_text.append('—')
        elif (current_token.startswith('<type_') or current_token.startswith('<subtype_')) and current_token.endswith('>'):
            type_line_text.append(current_token.replace('<type_', '').replace('<subtype_', '').replace('>', ''))
        else:
            raise ValueError(f"Malformed token stream: unexpected token {current_token!r} in type line block")
        current_token = token_stream.consume_token()
    return ' '.join(type_line_text)