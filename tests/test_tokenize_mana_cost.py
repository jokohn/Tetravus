import unittest
from tokenizers.tokenize_mana_cost import tokenize_mana_cost, detokenize_mana_cost
from special_tokens import begin_mana_cost_token, end_mana_cost_token
from token_stream import TokenStream

class TestTokenizeName(unittest.TestCase):
    def test_simple_mana_cost(self):
        mana_cost = '{1}{G}'

        tokens = tokenize_mana_cost(mana_cost)
        self.assertEqual(tokens, [begin_mana_cost_token, '<mana_cost_1>', '<mana_cost_G>', end_mana_cost_token])
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_mana_cost(token_stream), mana_cost)

    def test_phyrexian_mana(self):
        mana_cost = '{R/P}'

        tokens = tokenize_mana_cost(mana_cost)
        self.assertEqual(tokens, [begin_mana_cost_token, '<mana_cost_R/P>', end_mana_cost_token])
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_mana_cost(token_stream), mana_cost)
