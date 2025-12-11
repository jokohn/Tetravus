
def tokenize_release_year(release_year):
    return [f'<release_year_{release_year}>']

def detokenize_release_year(token_stream):
    release_year = token_stream.consume_token()
    return release_year.replace('<release_year_', '').replace('>', '')

def tokenize_rarity(rarity):
    return [f'<rarity_{rarity}>']

def detokenize_rarity(token_stream):
    rarity = token_stream.consume_token()
    return rarity.replace('<rarity_', '').replace('>', '')

def tokenize_set_name(set_name):
    return [f'<set_{set_name}>']

def detokenize_set_name(token_stream):
    set_name = token_stream.consume_token()
    return set_name.replace('<set_', '').replace('>', '')

def tokenize_power(power):
    return [f'<power_{power}>']

def detokenize_power(token_stream):
    power = token_stream.consume_token()
    return power.replace('<power_', '').replace('>', '')

def tokenize_toughness(toughness):
    return [f'<toughness_{toughness}>']

def detokenize_toughness(token_stream):
    toughness = token_stream.consume_token()
    return toughness.replace('<toughness_', '').replace('>', '')