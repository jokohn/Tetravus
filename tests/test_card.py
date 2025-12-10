import unittest
import json
import os

from card import Card

class TestCard(unittest.TestCase):
    def test_to_json(self):
      test_file = os.path.join(os.path.dirname(__file__), "test_data", "grizzly_bear.json")
      with open(test_file, "r") as f:
        data = json.load(f)
      card = Card.from_json(None ,data)
      self.assertEqual(card.to_json(), {
        "name": data["name"],
        "oracle_text": data.get("oracle_text", ""),
        "mana_cost": data["mana_cost"],
        "type_line": data["type_line"],
        "release_year": data["released_at"].split("-")[0],
        "rarity": data["rarity"],
        "set_name": data["set_name"]
      })

if __name__ == "__main__":
    unittest.main()