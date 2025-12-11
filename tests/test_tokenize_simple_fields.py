import unittest
import json
import os

from card import Card
from tokenizers.tokenize_simple_card_fields import tokenize_release_year, detokenize_release_year, tokenize_rarity, \
    detokenize_rarity, tokenize_set_name, detokenize_set_name,tokenize_power, detokenize_power, tokenize_toughness, \
    detokenize_toughness
from token_stream import TokenStream

class TestTokenizeSimpleFields(unittest.TestCase):
    def test_tokenize_release_year(self):
        release_year = "2025"
        tokens = tokenize_release_year(release_year)
        self.assertEqual(tokens, ['<release_year_2025>'])
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_release_year(token_stream), release_year)

    def test_tokenize_rarity(self):
        rarity = "common"
        tokens = tokenize_rarity(rarity)
        self.assertEqual(tokens, ['<rarity_common>'])
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_rarity(token_stream), rarity)
    
    def test_tokenize_set_name(self):
        set_code = "ZNR"
        tokens = tokenize_set_name(set_code)
        self.assertEqual(tokens, ['<set_ZNR>'])
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_set_name(token_stream), set_code)

    def tokenize_power(self):
        power = "4"
        tokens = tokenize_power(power)
        self.assertEqual(tokens, ['<power_4>'])
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_power(token_stream), power)

    def test_tokenize_toughness(self):
        toughness = "4"
        tokens = tokenize_toughness(toughness)
        self.assertEqual(tokens, ['<toughness_4>'])
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_toughness(token_stream), toughness)

    def test_tokenize_real_card(self):
        test_file = os.path.join(os.path.dirname(__file__), "test_data", "grizzly_bear.json")
        with open(test_file, "r") as f:
            data = json.load(f)
        card = Card.from_json(None, data)
        tokens = tokenize_release_year(card.release_year)
        self.assertEqual(tokens, ['<release_year_2007>'])
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_release_year(token_stream), card.release_year)
        tokens = tokenize_rarity(card.rarity)
        self.assertEqual(tokens, ['<rarity_common>'])
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_rarity(token_stream), card.rarity)
        tokens = tokenize_set_name(card.set_code)
        self.assertEqual(tokens, ['<set_10e>'])
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_set_name(token_stream), card.set_code)
        
        power = card.power
        tokens = tokenize_power(power)
        self.assertEqual(tokens, ['<power_2>'])
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_power(token_stream), power)
        toughness = card.toughness
        tokens = tokenize_toughness(toughness)
        self.assertEqual(tokens, ['<toughness_2>'])
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_toughness(token_stream), toughness)

if __name__ == "__main__":
    unittest.main()