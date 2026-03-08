from special_tokens import begin_mana_cost_token, end_mana_cost_token, begin_oracle_text_mana_cost_token, end_oracle_text_mana_cost_token
import re

def tokenize_mana_cost(mana_cost_string, is_orcale_text_mana_cost=False):
    """
    Tokenize a mana cost string in format '{char}+' (e.g., '{R}{G}{W}' or '{1}{2}{3}').
    
    Args:
        mana_cost_string: String in format '{char}+'
    
    Returns:
        List of tokens starting with begin_mana_cost_token, containing <mana_cost_{char}> 
        tokens for each character, and ending with end_mana_cost_token.
    """
    if is_orcale_text_mana_cost:
        tokens = [begin_oracle_text_mana_cost_token]
    else:
        tokens = [begin_mana_cost_token]
    
    # Extract characters from between braces using regex
    # Pattern matches {char} and captures the character
    pattern = r'\{([^}]+)\}'
    matches = re.findall(pattern, mana_cost_string)
    
    for char in matches:
        tokens.append(f'<{"oracle_text_" if is_orcale_text_mana_cost else ""}mana_cost_{char}>')
    
    if is_orcale_text_mana_cost:
        tokens.append(end_oracle_text_mana_cost_token)
    else:
        tokens.append(end_mana_cost_token)
    return tokens

def detokenize_mana_cost(token_stream, is_orcale_text_mana_cost=False):
    """
    Detokenize a token stream back into a mana cost string.
    
    Args:
        token_stream: TokenStream object containing mana cost tokens
    
    Returns:
        String in format '{char}+' (e.g., '{R}{G}{W}')
    """
    start_token = token_stream.peek()
    if is_orcale_text_mana_cost:
        if start_token != begin_oracle_text_mana_cost_token:
            raise ValueError(f"Expected begin_oracle_text_mana_cost_token, got {start_token}")
    else:
        if start_token != begin_mana_cost_token:
            raise ValueError(f"Expected begin_mana_cost_token, got {start_token}")
    token_stream.advance()
    mana_cost_chars = []
    
    end_token = end_oracle_text_mana_cost_token if is_orcale_text_mana_cost else end_mana_cost_token
    valid_prefix = '<oracle_text_mana_cost_' if is_orcale_text_mana_cost else '<mana_cost_'
    current_token = token_stream.consume_token()
    while current_token != end_token:
        if not (current_token.startswith(valid_prefix) and current_token.endswith('>')):
            raise ValueError(f"Malformed token stream: unexpected token {current_token!r} in mana cost block")
        char = current_token.replace('<mana_cost_', '').replace('>', '').replace('<oracle_text_mana_cost_', '')
        mana_cost_chars.append(f'{char.upper()}')
        current_token = token_stream.consume_token()
    
    # Reconstruct string in format {char}+
    return ''.join(f'{{{char}}}' for char in mana_cost_chars)