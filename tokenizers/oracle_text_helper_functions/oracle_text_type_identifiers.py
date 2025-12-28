import re

def is_stat_change_token(token_string):
    return re.match(r'^[\+\-][0-9a-zA-Z]+\/[\+\-][0-9a-zA-Z]+$', token_string) is not None

def is_stat_definition_token(token_string):
    if token_string == 'and/or':
        return False
    return re.match(r'^[0-9a-zA-Z]+\/[0-9a-zA-Z]+$', token_string) is not None

def is_planeswalker_loyalty_ability_token(token_string, type_line):
    if 'Planeswalker' not in (type_line or ''):
        return False
    return re.match(r'^(0|[\+−][1-9X]+)$', token_string) is not None

def is_planeswalker_loyaly_ability_colon_token(token_string, type_line):
    if 'Planeswalker' not in (type_line or ''):
        return False
    return token_string == ':'

def is_mana_cost_token(token_string):
    return re.match(r'^(\{[0-9a-zA-Z]+\}|\{[0-9a-zA-Z]\/[0-9a-zA-Z]\})+$', token_string)
