from special_tokens import begin_planeswalker_loyalty_ability_cost_token, end_planeswalker_loyalty_ability_cost_token

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
    if not current_token.startswith('<planeswalker_loyalty_ability_cost_sign_'):
        raise ValueError(f"Expected <planeswalker_loyalty_ability_cost_sign_>, got {current_token}")
    sign = current_token.replace('<planeswalker_loyalty_ability_cost_sign_', '').replace('>', '')
    current_token = token_stream.consume_token()
    if not current_token.startswith('<planeswalker_loyalty_ability_cost_value_'):
        raise ValueError(f"Expected <planeswalker_loyalty_ability_cost_value_>, got {current_token}")
    value = current_token.replace('<planeswalker_loyalty_ability_cost_value_', '').replace('>', '')
    current_token = token_stream.consume_token()
    if current_token != end_planeswalker_loyalty_ability_cost_token:
        raise ValueError(f"Expected end_planeswalker_loyalty_ability_cost_token, got {current_token}")
    return f'{sign}{value}:'    