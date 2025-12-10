
class TokenStream:
    def __init__(self, tokens):
        self.tokens = tokens
        self.index = 0

    def has_next(self):
        return self.index < len(self.tokens)

    def consume_token(self):
        if not self.has_next():
            raise IndexError("No more tokens to consume")
        token = self.tokens[self.index]
        self.advance()
        return token
    
    def peek(self):
        return self.tokens[self.index]

    def advance(self):
        self.index += 1

    def jump_by(self, jump_amount):
        self.index += jump_amount

    def jump_to(self, index):
        if index < 0 or index >= len(self.tokens):
            raise IndexError(f"Index out of range: {index}")
        self.index = index

    def reset(self):
        self.index = 0