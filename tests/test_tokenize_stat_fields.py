import unittest

from tokenizers.oracle_text_helper_functions.tokenize_special_oracle_text_fields import tokenize_stats_definition_string, \
    detokenize_stats_definition_string, tokenize_stats_change_string, detokenize_stats_change_string
from special_tokens import begin_stats_definition_token, end_stats_definition_token, begin_stats_change_token, \
    end_stats_change_token
from token_stream import TokenStream

class TestTokenizeStatFields(unittest.TestCase):
    def test_tokenize_stats_definition_string(self):
        token_string = '4/4'
        tokens = tokenize_stats_definition_string(token_string)
        self.assertEqual(tokens, [begin_stats_definition_token, '<stats_definition_power_4>', '<stats_definition_toughness_4>', end_stats_definition_token])
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_stats_definition_string(token_stream), token_string)

    def tokenize_non_numeric_stats_definition_string(self):
        token_string = 'X/X'
        tokens = tokenize_stats_definition_string(token_string)
        self.assertEqual(tokens, [begin_stats_definition_token, '<stats_definition_power_X>', '<stats_definition_toughness_X>', end_stats_definition_token])
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_stats_definition_string(token_stream), token_string)

    def tokenize_multi_digit_stats_definition_string(self):
        token_string = '10/10'
        tokens = tokenize_stats_definition_string(token_string)
        self.assertEqual(tokens, [begin_stats_definition_token, '<stats_definition_power_10>', '<stats_definition_toughness_10>', end_stats_definition_token])
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_stats_definition_string(token_stream), token_string)

    def test_tokenize_stats_change_string(self):
        token_string = '+2/+2'
        tokens = tokenize_stats_change_string(token_string)
        self.assertEqual(
            tokens, [
                begin_stats_change_token,
                '<stats_change_power_sign_+>',
                '<stats_change_power_value_2>',
                '<stats_change_toughness_sign_+>',
                '<stats_change_toughness_value_2>',
                end_stats_change_token
            ]
        )
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_stats_change_string(token_stream), token_string)

    def tokenize_negative_stats_change_string(self):
        token_string = '-2/-2'
        tokens = tokenize_stats_change_string(token_string)
        self.assertEqual(
            tokens, [
                begin_stats_change_token,
                '<stats_change_power_sign_->',
                '<stats_change_power_value_2>',
                '<stats_change_toughness_sign_->',
                '<stats_change_toughness_value_2>',
                end_stats_change_token
            ]
        )
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_stats_change_string(token_stream), token_string)

    def test_non_numeric_stats_change_string(self):
        token_string = '+X/+X'
        tokens = tokenize_stats_change_string(token_string)
        self.assertEqual(
            tokens, [
                begin_stats_change_token,
                '<stats_change_power_sign_+>',
                '<stats_change_power_value_X>',
                '<stats_change_toughness_sign_+>',
                '<stats_change_toughness_value_X>',
                end_stats_change_token
            ]
        )
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_stats_change_string(token_stream), token_string)

    def test_detokenize_stats_definition_rejects_wrong_block_token(self):
        """Malformed stream: name_char token in stats_definition block raises ValueError."""
        malformed = [begin_stats_definition_token, "<name_char_4>", "<stats_definition_toughness_4>", end_stats_definition_token]
        with self.assertRaises(ValueError) as ctx:
            detokenize_stats_definition_string(TokenStream(malformed))
        self.assertIn("Malformed token stream", str(ctx.exception))
        self.assertIn("stats_definition", str(ctx.exception))

    def test_detokenize_stats_change_rejects_wrong_block_token(self):
        """Malformed stream: oracle_text token in stats_change block raises ValueError."""
        malformed = [
            begin_stats_change_token,
            "<stats_change_power_sign_+>",
            "<oracle_text_2>",
            "<stats_change_toughness_sign_+>",
            "<stats_change_toughness_value_2>",
            end_stats_change_token,
        ]
        with self.assertRaises(ValueError) as ctx:
            detokenize_stats_change_string(TokenStream(malformed))
        self.assertIn("Malformed token stream", str(ctx.exception))
        self.assertIn("stats_change", str(ctx.exception))