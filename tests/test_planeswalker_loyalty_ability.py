import os
import json
import unittest
from tokenizers.oracle_text_helper_functions.tokenize_special_oracle_text_fields import tokenize_planeswalker_loyalty_ability, detokenize_planeswalker_loyalty_ability
from special_tokens import begin_planeswalker_loyalty_ability_cost_token, end_planeswalker_loyalty_ability_cost_token
from token_stream import TokenStream
from card import Card
from tokenizers.tokenize_oracle_text import detokenize_oracle_text

class TestPlaneswalkerLoyaltyAbility(unittest.TestCase):
    def test_tokenize_planeswalker_loyalty_ability_0(self):
        token_string = '0'
        tokens = tokenize_planeswalker_loyalty_ability(token_string)
        self.assertEqual(tokens, [begin_planeswalker_loyalty_ability_cost_token, '<planeswalker_loyalty_ability_cost_value_0>', end_planeswalker_loyalty_ability_cost_token])
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_planeswalker_loyalty_ability(token_stream), '0:')

    def test_tokenize_planeswalker_loyalty_ability_positive(self):
        token_string = '+1'
        tokens = tokenize_planeswalker_loyalty_ability(token_string)
        self.assertEqual(tokens, [begin_planeswalker_loyalty_ability_cost_token, '<planeswalker_loyalty_ability_cost_sign_+>', '<planeswalker_loyalty_ability_cost_value_1>', end_planeswalker_loyalty_ability_cost_token])
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_planeswalker_loyalty_ability(token_stream), '+1:')

    def test_tokenize_planeswalker_loyalty_ability_negative(self):
        token_string = '-1'
        tokens = tokenize_planeswalker_loyalty_ability(token_string)
        self.assertEqual(tokens, [begin_planeswalker_loyalty_ability_cost_token, '<planeswalker_loyalty_ability_cost_sign_->', '<planeswalker_loyalty_ability_cost_value_1>', end_planeswalker_loyalty_ability_cost_token])
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_planeswalker_loyalty_ability(token_stream), '-1:')

    def test_tokenize_planeswalker_loyalty_ability_positive_X(self):
        token_string = '+X'
        tokens = tokenize_planeswalker_loyalty_ability(token_string)
        self.assertEqual(tokens, [begin_planeswalker_loyalty_ability_cost_token, '<planeswalker_loyalty_ability_cost_sign_+>', '<planeswalker_loyalty_ability_cost_value_X>', end_planeswalker_loyalty_ability_cost_token])
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_planeswalker_loyalty_ability(token_stream), '+X:')

    def test_ob_nixilis_hate_twisted(self):
        test_file = os.path.join(os.path.dirname(__file__), "test_data", "ob_nixilis_hate_twisted.json")
        with open(test_file, "r", encoding='utf-8') as f:
            data = json.load(f)
        card = Card.from_json(None, data)
        tokens = card.generate_tokens(["oracle_text"])
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_oracle_text(token_stream, card_name=card.name), card.oracle_text.replace('Ob Nixilis', 'Ob Nixilis, the Hate-Twisted'))

    def test_detokenize_planeswalker_loyalty_ability_rejects_wrong_block_token(self):
        """Malformed stream: name_char token in planeswalker_loyalty_ability block raises ValueError."""
        malformed = [
            begin_planeswalker_loyalty_ability_cost_token,
            "<name_char_+>",
            "<planeswalker_loyalty_ability_cost_value_1>",
            end_planeswalker_loyalty_ability_cost_token,
        ]
        with self.assertRaises(ValueError) as ctx:
            detokenize_planeswalker_loyalty_ability(TokenStream(malformed))
        self.assertIn("Malformed token stream", str(ctx.exception))
        self.assertIn("planeswalker_loyalty_ability", str(ctx.exception))