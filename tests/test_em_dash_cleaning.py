import unittest

import os
import json

from card import Card
from special_tokens import begin_oracle_text_token, end_oracle_text_token, \
    begin_stats_change_token, end_stats_change_token
from token_stream import TokenStream
from tokenizers.tokenize_oracle_text import detokenize_oracle_text, tokenize_oracle_text

class TestEmDashCleaning(unittest.TestCase):

    def test_safe_word(self):
        oracle_text = "Cumulative upkeep—Pay 1 life."

        tokens = tokenize_oracle_text(oracle_text)
        self.assertEqual(tokens, [
            begin_oracle_text_token,
            '<oracle_text_cumulative>',
            '<oracle_text_upkeep>',
            '<oracle_text_—>',
            '<oracle_text_pay>',
            '<oracle_text_1>',
            '<oracle_text_life>',
            '<oracle_text_.>',
            end_oracle_text_token
        ])
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_oracle_text(token_stream), oracle_text.replace('—', ' — ').replace('Pay', 'pay'))

    def test_blocked_word(self):
        oracle_text = "Flavor Keyword—Pay 1 life."

        tokens = tokenize_oracle_text(oracle_text)
        self.assertEqual(tokens, [
            begin_oracle_text_token,
            '<oracle_text_pay>',
            '<oracle_text_1>',
            '<oracle_text_life>',
            '<oracle_text_.>',
            end_oracle_text_token
        ])
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_oracle_text(token_stream), oracle_text.replace('Flavor Keyword—', ''))


    def test_you_come_to_a_river(self):
        test_file = os.path.join(os.path.dirname(__file__), "test_data", "you_come_to_a_river.json")
        with open(test_file, "r") as f:
            data = json.load(f)
        card = Card.from_json(None, data)
        tokens = card.generate_tokens(["oracle_text"])
        self.assertEqual(tokens, [
            begin_oracle_text_token,
            '<oracle_text_choose>',
            '<oracle_text_one>',
            '<oracle_text_—>',
            '<oracle_text_\n>',
            '<oracle_text_•>',
            '<oracle_text_return>',
            '<oracle_text_target>',
            '<oracle_text_nonland>',
            '<oracle_text_permanent>',
            '<oracle_text_to>',
            '<oracle_text_its>',
            "<oracle_text_owner's>",
            '<oracle_text_hand>',
            '<oracle_text_.>',
            '<oracle_text_\n>',
            '<oracle_text_•>',
            '<oracle_text_target>',
            '<oracle_text_creature>',
            '<oracle_text_gets>',
            begin_stats_change_token,
            '<stats_change_power_sign_+>',
            '<stats_change_power_value_1>',
            '<stats_change_toughness_sign_+>',
            '<stats_change_toughness_value_0>',
            end_stats_change_token,
            '<oracle_text_until>',
            '<oracle_text_end>',
            '<oracle_text_of>',
            '<oracle_text_turn>',
            '<oracle_text_and>',
            "<oracle_text_can't>",
            '<oracle_text_be>',
            '<oracle_text_blocked>',
            '<oracle_text_this>',
            '<oracle_text_turn>',
            '<oracle_text_.>',
            end_oracle_text_token
        ])
        token_stream = TokenStream(tokens)
        self.assertEqual(
            detokenize_oracle_text(token_stream).lower(),
            card.oracle_text.lower().replace('fight the current — ', '').replace('find a crossing — ', '')
        )