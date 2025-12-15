import unittest
import json
import os

from tokenize_card_file import tokenize_card_file
from special_tokens import begin_name_token

class TestTokenizeCardFile(unittest.TestCase):
    def test_tokenize_card_file(self):
        test_file = os.path.join(os.path.dirname(__file__), "test_data", "cleaned_cards.json")
        token_map, token_string, token_metadata, processed_cards, failed_cards, token_count = tokenize_card_file(test_file)
        self.assertEqual(4, token_metadata[begin_name_token])
        self.assertEqual(4, processed_cards)
        self.assertEqual(1, failed_cards)
        self.assertEqual(162, token_count)