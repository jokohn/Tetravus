import unittest
import json
import os

from card import Card
from tokenizers.tokenize_type_line import tokenize_type_line, detokenize_type_line
from special_tokens import begin_type_line_token, end_type_line_token, subtype_break_token
from token_stream import TokenStream

class TestTokenizeTypeLine(unittest.TestCase):
    def test_tokenize_type_line(self):
        type_line_text = "Creature — Bear"
        tokens = tokenize_type_line(type_line_text)
        self.assertEqual(tokens, [begin_type_line_token, '<type_Creature>', subtype_break_token, '<subtype_Bear>', end_type_line_token])
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_type_line(token_stream), type_line_text)

    def test_tokenize_type_line_with_multiple_types(self):
        type_line_text = "Legendary Creature — Bear Noble"
        tokens = tokenize_type_line(type_line_text)
        self.assertEqual(tokens, [begin_type_line_token, '<type_Legendary>', '<type_Creature>', subtype_break_token, '<subtype_Bear>', '<subtype_Noble>', end_type_line_token])
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_type_line(token_stream), type_line_text)

    def test_no_subtypes(self):
        type_line_text = "Legendary Artifact"
        tokens = tokenize_type_line(type_line_text)
        self.assertEqual(tokens, [begin_type_line_token, '<type_Legendary>', '<type_Artifact>', end_type_line_token])
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_type_line(token_stream), type_line_text)

    def test_tokenize_real_card(self):
        test_file = os.path.join(os.path.dirname(__file__), "test_data", "grizzly_bear.json")
        with open(test_file, "r", encoding='utf-8') as f:
            data = json.load(f)
        card = Card.from_json(None ,data)
        tokens = tokenize_type_line(card.type_line)
        self.assertEqual(tokens, [begin_type_line_token, '<type_Creature>', subtype_break_token, '<subtype_Bear>', end_type_line_token])
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_type_line(token_stream), card.type_line)

    def test_detokenize_type_line_rejects_wrong_block_token(self):
        """Malformed stream: name_char token in type line block raises ValueError."""
        malformed = [begin_type_line_token, "<name_char_C>", end_type_line_token]
        with self.assertRaises(ValueError) as ctx:
            detokenize_type_line(TokenStream(malformed))
        self.assertIn("Malformed token stream", str(ctx.exception))
        self.assertIn("type line block", str(ctx.exception))
        self.assertIn("<name_char_C>", str(ctx.exception))

if __name__ == "__main__":
    unittest.main()