import unittest
from tokenizers.tokenize_name import tokenize_name, detokenize_name
from token_stream import TokenStream

class TestTokenizeName(unittest.TestCase):
    def test_tokenize_name(self):
        name = "Grizzly Bear"
        tokens = tokenize_name(name)
        self.assertEqual(tokens, [
            "<name>",
            "<name_char_G>",
            "<name_char_r>",
            "<name_char_i>",
            "<name_char_z>",
            "<name_char_z>",
            "<name_char_l>",
            "<name_char_y>",
            "<name_char_ >",
            "<name_char_B>",
            "<name_char_e>",
            "<name_char_a>",
            "<name_char_r>",
            "</name>"
        ])
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_name(token_stream), name)

if __name__ == "__main__":
    unittest.main()