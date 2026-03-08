from special_tokens import begin_planeswalker_loyalty_ability_cost_token, end_planeswalker_loyalty_ability_cost_token, \
    begin_stats_definition_token, end_stats_definition_token, begin_stats_change_token, end_stats_change_token

# Prefixes that indicate a token belongs to another block (for malformed-stream detection)
_OTHER_BLOCK_PREFIXES = (
    '<name_char_', '<oracle_text_', '<type_', '<subtype_', '<mana_cost_',
    '<release_year_', '<rarity_', '<set_', '<power_', '<toughness_', '<loyalty_',
)


def _raise_if_other_block_token(token, block_name):
    """Raise ValueError with malformed message if token has another block's prefix."""
    for prefix in _OTHER_BLOCK_PREFIXES:
        if token.startswith(prefix):
            raise ValueError(f"Malformed token stream: unexpected token {token!r} in {block_name} block")


def tokenize_planeswalker_loyalty_ability(token_string):
    if token_string.startswith('0'):
        return [begin_planeswalker_loyalty_ability_cost_token, '<planeswalker_loyalty_ability_cost_value_0>', end_planeswalker_loyalty_ability_cost_token]
    else:
        return [
            begin_planeswalker_loyalty_ability_cost_token,
            f'<planeswalker_loyalty_ability_cost_sign_{token_string[0]}>',
            f'<planeswalker_loyalty_ability_cost_value_{token_string[1:]}>',
            end_planeswalker_loyalty_ability_cost_token,
            ]

def detokenize_planeswalker_loyalty_ability(token_stream):
    start_token = token_stream.peek()
    if start_token != begin_planeswalker_loyalty_ability_cost_token:
        raise ValueError(f"Expected begin_planeswalker_loyalty_ability_cost_token, got {start_token}")
    token_stream.advance()
    current_token = token_stream.consume_token()

    if current_token == '<planeswalker_loyalty_ability_cost_value_0>':
        token_stream.advance()
        return '0:'
    _raise_if_other_block_token(current_token, 'planeswalker_loyalty_ability')
    if not current_token.startswith('<planeswalker_loyalty_ability_cost_sign_'):
        raise ValueError(f"Expected <planeswalker_loyalty_ability_cost_sign_>, got {current_token}")
    sign = current_token.replace('<planeswalker_loyalty_ability_cost_sign_', '').replace('>', '')
    current_token = token_stream.consume_token()
    _raise_if_other_block_token(current_token, 'planeswalker_loyalty_ability')
    if not current_token.startswith('<planeswalker_loyalty_ability_cost_value_'):
        raise ValueError(f"Expected <planeswalker_loyalty_ability_cost_value_>, got {current_token}")
    value = current_token.replace('<planeswalker_loyalty_ability_cost_value_', '').replace('>', '')
    current_token = token_stream.consume_token()
    _raise_if_other_block_token(current_token, 'planeswalker_loyalty_ability')
    if current_token != end_planeswalker_loyalty_ability_cost_token:
        raise ValueError(f"Expected end_planeswalker_loyalty_ability_cost_token, got {current_token}")
    return f'{sign}{value}:'    

def tokenize_stats_definition_string(token_string):
    power_change, toughness_change = token_string.split('/')
    tokens = [begin_stats_definition_token]
    tokens.append(f'<stats_definition_power_{power_change}>')
    tokens.append(f'<stats_definition_toughness_{toughness_change}>')
    tokens.append(end_stats_definition_token)
    return tokens

def detokenize_stats_definition_string(token_stream):
    start_token = token_stream.peek()
    if start_token != begin_stats_definition_token:
        raise ValueError(f"Expected begin_stats_definition_token, got {start_token}")
    token_stream.advance()
    current_token = token_stream.consume_token()
    _raise_if_other_block_token(current_token, 'stats_definition')
    if not current_token.startswith('<stats_definition_power_'):
        raise ValueError(f"Expected <stats_definition_power_>, got {current_token}")
    power = current_token.replace('<stats_definition_power_', '').replace('>', '')
    current_token = token_stream.consume_token()
    _raise_if_other_block_token(current_token, 'stats_definition')
    if not current_token.startswith('<stats_definition_toughness_'):
        raise ValueError(f"Expected <stats_definition_toughness_>, got {current_token}")
    toughness = current_token.replace('<stats_definition_toughness_', '').replace('>', '')
    current_token = token_stream.consume_token()
    _raise_if_other_block_token(current_token, 'stats_definition')
    if current_token != end_stats_definition_token:
        raise ValueError(f"Expected end_stats_definition_token, got {current_token}")
    return f'{power}/{toughness}'

def tokenize_stats_change_string(token_string):
    power_change, toughness_change = token_string.split('/')
    tokens = [begin_stats_change_token]
    if power_change.startswith('+'):
        tokens.append('<stats_change_power_sign_+>')
    elif power_change.startswith('-'):
        tokens.append('<stats_change_power_sign_->')
    else:
        raise ValueError(f"Expected string to start with + or -, got {power_change}")
    tokens.append(f'<stats_change_power_value_{power_change[1:]}>')
    if toughness_change.startswith('+'):
        tokens.append('<stats_change_toughness_sign_+>')
    elif toughness_change.startswith('-'):
        tokens.append('<stats_change_toughness_sign_->')
    else:
        raise ValueError(f"Expected string to start with + or -, got {toughness_change}")
    tokens.append(f'<stats_change_toughness_value_{toughness_change[1:]}>')
    tokens.append(end_stats_change_token)
    return tokens

def detokenize_stats_change_string(token_stream):
    start_token = token_stream.peek()
    if start_token != begin_stats_change_token:
        raise ValueError(f"Expected begin_stats_change_token, got {start_token}")
    token_stream.advance()
    current_token = token_stream.consume_token()
    _raise_if_other_block_token(current_token, 'stats_change')
    if not current_token.startswith('<stats_change_power_sign_'):
        raise ValueError(f"Expected <stats_change_power_sign_>, got {current_token}")
    power_sign = current_token.replace('<stats_change_power_sign_', '').replace('>', '')
    current_token = token_stream.consume_token()
    _raise_if_other_block_token(current_token, 'stats_change')
    if not current_token.startswith('<stats_change_power_value_'):
        raise ValueError(f"Expected <stats_change_power_value_>, got {current_token}")
    power_value = current_token.replace('<stats_change_power_value_', '').replace('>', '')
    current_token = token_stream.consume_token()
    _raise_if_other_block_token(current_token, 'stats_change')
    if not current_token.startswith('<stats_change_toughness_sign_'):
        raise ValueError(f"Expected <stats_change_toughness_sign_>, got {current_token}")
    toughness_sign = current_token.replace('<stats_change_toughness_sign_', '').replace('>', '')
    current_token = token_stream.consume_token()
    _raise_if_other_block_token(current_token, 'stats_change')
    if not current_token.startswith('<stats_change_toughness_value_'):
        raise ValueError(f"Expected <stats_change_toughness_value_>, got {current_token}")
    toughness_value = current_token.replace('<stats_change_toughness_value_', '').replace('>', '')
    current_token = token_stream.consume_token()
    _raise_if_other_block_token(current_token, 'stats_change')
    if current_token != end_stats_change_token:
        raise ValueError(f"Expected end_stats_change_token, got {current_token}")
    return f'{power_sign}{power_value}/{toughness_sign}{toughness_value}'