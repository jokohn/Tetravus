import unittest
import json
import os

from card_fetcher import card_cleaner

class TestCardCleaner(unittest.TestCase):
    def test_card_cleaner(self):
        test_file = os.path.join(os.path.dirname(__file__), "test_data", "raw_cards.json")
        with open(test_file, "r") as f:
            data = json.load(f)
        cards = card_cleaner(test_file)
        self.assertEqual(len(cards), 3)

        self.assertEqual(cards[1]['name'], "Fury Sliver")