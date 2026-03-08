
def tokenize_release_year(release_year):
    return [f'<release_year_{release_year}>']

def detokenize_release_year(token_stream):
    token = token_stream.consume_token()
    if not (token.startswith('<release_year_') and token.endswith('>')):
        raise ValueError(f"Malformed token stream: unexpected token {token!r} in release_year block")
    return token.replace('<release_year_', '').replace('>', '')

def tokenize_rarity(rarity):
    return [f'<rarity_{rarity}>']

def detokenize_rarity(token_stream):
    token = token_stream.consume_token()
    if not (token.startswith('<rarity_') and token.endswith('>')):
        raise ValueError(f"Malformed token stream: unexpected token {token!r} in rarity block")
    return token.replace('<rarity_', '').replace('>', '')

def tokenize_set_name(set_name):
    return [f'<set_{set_name}>']

def detokenize_set_name(token_stream):
    token = token_stream.consume_token()
    if not (token.startswith('<set_') and token.endswith('>')):
        raise ValueError(f"Malformed token stream: unexpected token {token!r} in set_name block")
    return token.replace('<set_', '').replace('>', '')

def tokenize_power(power):
    if not power:
        return []
    return [f'<power_{power}>']

def detokenize_power(token_stream):
    token = token_stream.consume_token()
    if not (token.startswith('<power_') and token.endswith('>')):
        raise ValueError(f"Malformed token stream: unexpected token {token!r} in power block")
    return token.replace('<power_', '').replace('>', '')

def tokenize_toughness(toughness):
    if not toughness:
        return []
    return [f'<toughness_{toughness}>']

def detokenize_toughness(token_stream):
    token = token_stream.consume_token()
    if not (token.startswith('<toughness_') and token.endswith('>')):
        raise ValueError(f"Malformed token stream: unexpected token {token!r} in toughness block")
    return token.replace('<toughness_', '').replace('>', '')

def tokenize_loyalty(loyalty):
    if not loyalty:
        return []
    return [f'<loyalty_{loyalty}>']

def detokenize_loyalty(token_stream):
    token = token_stream.consume_token()
    if not (token.startswith('<loyalty_') and token.endswith('>')):
        raise ValueError(f"Malformed token stream: unexpected token {token!r} in loyalty block")
    return token.replace('<loyalty_', '').replace('>', '')