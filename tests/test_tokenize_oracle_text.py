import unittest
from re import sub
import json
import os

from card import Card
from tokenizers.tokenize_oracle_text import tokenize_oracle_text, detokenize_oracle_text
from token_stream import TokenStream
from tokenizers.oracle_text_helper_functions.preprocess_oracle_text import UnsupportedCharacterError

class TestTokenizeOracleText(unittest.TestCase):
    def test_tokenize_oracle_text_simple(self):
        oracle_text = "Draw three cards."
        tokens = tokenize_oracle_text(oracle_text)
        self.assertEqual(tokens, [
            "<oracle_text>",
            "<oracle_text_draw>",
            "<oracle_text_three>",
            "<oracle_text_cards>",
            "<oracle_text_.>",
            "</oracle_text>"
        ])
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_oracle_text(token_stream), oracle_text)

    def test_strip_reminder_text(self):
        oracle_text = "Create a blood token. (It's an artifact with \"{1}, {T}, Discard a card, Sacrifice this token: Draw a card.\")"
        tokens = tokenize_oracle_text(oracle_text)
        self.assertEqual(tokens, [
            "<oracle_text>",
            "<oracle_text_create>",
            "<oracle_text_a>",
            "<oracle_text_blood>",
            "<oracle_text_token>",
            "<oracle_text_.>",
            "</oracle_text>"
        ])
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_oracle_text(token_stream), 'Create a blood token.')

    def test_newline_characters(self):
        oracle_text = "Draw a card.\nGain 3 life. Win the game."
        tokens = tokenize_oracle_text(oracle_text)
        self.assertEqual(tokens, [
            "<oracle_text>",
            "<oracle_text_draw>",
            "<oracle_text_a>",
            "<oracle_text_card>",
            "<oracle_text_.>",
            "<oracle_text_\n>",
            "<oracle_text_gain>",
            "<oracle_text_3>",
            "<oracle_text_life>",
            "<oracle_text_.>",
            "<oracle_text_win>",
            "<oracle_text_the>",
            "<oracle_text_game>",
            "<oracle_text_.>",
            "</oracle_text>"
        ])
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_oracle_text(token_stream), oracle_text)

    def test_aang_air_nomad(self):
        test_file = os.path.join(os.path.dirname(__file__), "test_data", "aang_air_nomad.json")
        with open(test_file, "r") as f:
            data = json.load(f)
        card = Card.from_json(None, data)
        tokens = tokenize_oracle_text(card.oracle_text)
        self.assertEqual(tokens, [
            "<oracle_text>",
            "<oracle_text_flying>",
            "<oracle_text_\n>",
            "<oracle_text_vigilance>",
            "<oracle_text_\n>",
            "<oracle_text_other>",
            "<oracle_text_creatures>",
            "<oracle_text_you>",
            "<oracle_text_control>",
            "<oracle_text_have>",
            "<oracle_text_vigilance>",
            "<oracle_text_.>",
            "</oracle_text>"
        ])
        token_stream = TokenStream(tokens)
        self.assertEqual(
            detokenize_oracle_text(token_stream),
            'Flying\nVigilance\nOther creatures you control have vigilance.')

    def test_grizzly_bear(self):
        test_file = os.path.join(os.path.dirname(__file__), "test_data", "grizzly_bear.json")
        with open(test_file, "r") as f:
            data = json.load(f)
        card = Card.from_json(None, data)
        tokens = tokenize_oracle_text(card.oracle_text)
        self.assertEqual(tokens, [
            "<oracle_text>",
            "</oracle_text>"
        ])
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_oracle_text(token_stream), card.oracle_text)

    def test_unsupported_characters(self):
        oracle_text = "{U}: Mill a card."
        with self.assertRaises(UnsupportedCharacterError):
            tokenize_oracle_text(oracle_text)
