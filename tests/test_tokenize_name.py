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

    def test_detokenize_name_rejects_wrong_block_token(self):
        """Malformed stream: oracle_text token inside name block raises ValueError."""
        malformed = ["<name>", "<oracle_text_draw>", "</name>"]
        with self.assertRaises(ValueError) as ctx:
            detokenize_name(TokenStream(malformed))
        self.assertIn("Malformed token stream", str(ctx.exception))
        self.assertIn("name block", str(ctx.exception))
        self.assertIn("<oracle_text_draw>", str(ctx.exception))

if __name__ == "__main__":
    unittest.main()