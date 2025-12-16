
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
    if not power:
        return []
    return [f'<power_{power}>']

def detokenize_power(token_stream):
    power = token_stream.consume_token()
    return power.replace('<power_', '').replace('>', '')

def tokenize_toughness(toughness):
    if not toughness:
        return []
    return [f'<toughness_{toughness}>']

def detokenize_toughness(token_stream):
    toughness = token_stream.consume_token()
    return toughness.replace('<toughness_', '').replace('>', '')

def tokenize_loyalty(loyalty):
    if not loyalty:
        return []
    return [f'<loyalty_{loyalty}>']

def detokenize_loyalty(token_stream):
    loyalty = token_stream.consume_token()
    return loyalty.replace('<loyalty_', '').replace('>', '')