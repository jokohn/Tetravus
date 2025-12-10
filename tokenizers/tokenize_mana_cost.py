
from special_tokens import begin_mana_cost_token, end_mana_cost_token
import re

def tokenize_mana_cost(mana_cost_string):
    """
    Tokenize a mana cost string in format '{char}+' (e.g., '{R}{G}{W}' or '{1}{2}{3}').
    
    Args:
        mana_cost_string: String in format '{char}+'
    
    Returns:
        List of tokens starting with begin_mana_cost_token, containing <mana_cost_{char}> 
        tokens for each character, and ending with end_mana_cost_token.
    """
    tokens = [begin_mana_cost_token]
    
    # Extract characters from between braces using regex
    # Pattern matches {char} and captures the character
    pattern = r'\{([^}]+)\}'
    matches = re.findall(pattern, mana_cost_string)
    
    for char in matches:
        tokens.append(f'<mana_cost_{char}>')
    
    tokens.append(end_mana_cost_token)
    return tokens

def detokenize_mana_cost(token_stream):
    """
    Detokenize a token stream back into a mana cost string.
    
    Args:
        token_stream: TokenStream object containing mana cost tokens
    
    Returns:
        String in format '{char}+' (e.g., '{R}{G}{W}')
    """
    start_token = token_stream.peek()
    if start_token != begin_mana_cost_token:
        raise ValueError(f"Expected begin_mana_cost_token, got {start_token}")
    
    token_stream.advance()
    mana_cost_chars = []
    
    current_token = token_stream.consume_token()
    while current_token != end_mana_cost_token:
        # Extract character from token like <mana_cost_R>
        char = current_token.replace('<mana_cost_', '').replace('>', '')
        mana_cost_chars.append(f'{char}')
        current_token = token_stream.consume_token()
    
    # Reconstruct string in format {char}+
    return ''.join(f'{{{char}}}' for char in mana_cost_chars)