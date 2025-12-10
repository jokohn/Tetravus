import unittest

from token_stream import TokenStream

class TestTokenStream(unittest.TestCase):
    def test_token_stream(self):
        tokens = ["<name>", "<name_char_G>", "<name_char_r>", "<name_char_i>", "<name_char_z>", "<name_char_z>", "<name_char_y>", "<name_char_B>", "<name_char_e>", "<name_char_a>", "<name_char_r>", "</name>"]
        token_stream = TokenStream(tokens)
        self.assertTrue(token_stream.has_next())
        self.assertEqual(token_stream.peek(), "<name>")
        self.assertEqual(token_stream.consume_token(), "<name>")
        self.assertEqual(token_stream.peek(), "<name_char_G>")
        token_stream.advance()
        self.assertEqual(token_stream.peek(), "<name_char_r>")
        token_stream.jump_by(0)
        self.assertEqual(token_stream.peek(), "<name_char_r>")
        token_stream.jump_by(2)
        self.assertEqual(token_stream.peek(), "<name_char_z>")
        token_stream.jump_to(7)
        self.assertEqual(token_stream.peek(), "<name_char_B>")
        token_stream.reset()
        self.assertEqual(token_stream.peek(), "<name>")
        with self.assertRaises(IndexError):
            token_stream.jump_to(100)
        with self.assertRaises(IndexError):
            token_stream.jump_to(-1)
        with self.assertRaises(IndexError):
            token_stream.jump_to(len(tokens))
            token_stream.consume_token()


if __name__ == "__main__":
    unittest.main()