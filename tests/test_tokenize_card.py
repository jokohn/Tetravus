import unittest
import json
import os

from card import Card

from special_tokens import begin_name_token, end_name_token, begin_oracle_text_token, end_oracle_text_token, \
    begin_mana_cost_token, end_mana_cost_token, begin_type_line_token, subtype_break_token, end_type_line_token
from tokenizers.tokenize_oracle_text import tokenize_oracle_text
from tokenizers.tokenize_mana_cost import tokenize_mana_cost
from tokenizers.tokenize_name import tokenize_name
from tokenizers.tokenize_type_line import tokenize_type_line
from tokenizers.tokenize_simple_card_fields import tokenize_power, tokenize_toughness, tokenize_release_year, \
    tokenize_rarity, tokenize_set_name, tokenize_loyalty

class TestTokenizeCard(unittest.TestCase):
    def test_tokenize_card(self):
        test_file = os.path.join(os.path.dirname(__file__), "test_data", "grizzly_bear.json")
        with open(test_file, "r", encoding='utf-8') as f:
            data = json.load(f)
        card = Card.from_json(None, data)
        tokens = card.generate_tokens(["name", "oracle_text", "mana_cost", "type_line", "release_year", "rarity", "set", "power", "toughness"])
        self.assertEqual(
            ''.join(tokens),
                begin_name_token +
                    '<name_char_G><name_char_r><name_char_i><name_char_z><name_char_z><name_char_l><name_char_y>' +
                    '<name_char_ >' +
                    '<name_char_B><name_char_e><name_char_a><name_char_r>' +
                end_name_token +
                begin_oracle_text_token +
                end_oracle_text_token +
                begin_mana_cost_token +
                    '<mana_cost_1><mana_cost_G>' +
                end_mana_cost_token +
                begin_type_line_token +
                    '<type_Creature>' +
                    subtype_break_token +
                    '<subtype_Bear>' +
                end_type_line_token +
                '<release_year_2007>' +
                '<rarity_common>' +
                '<set_10e>' +
                '<power_2>' +
                '<toughness_2>'
        )

    def test_card_with_oracle_text(self):
        test_file = os.path.join(os.path.dirname(__file__), "test_data", "aang_air_nomad.json")
        with open(test_file, "r") as f:
            data = json.load(f)
        card = Card.from_json(None, data)
        tokens = card.generate_tokens(["name", "mana_cost", "oracle_text", "release_year", "rarity", "set", "power", "toughness", "type_line"])
        self.assertEqual(
            ''.join(tokens),
            ''.join(tokenize_name(card.name)) +
            ''.join(tokenize_mana_cost(card.mana_cost)) +
            ''.join(tokenize_oracle_text(card.oracle_text)) +
            ''.join(tokenize_release_year(card.release_year)) +
            ''.join(tokenize_rarity(card.rarity)) +
            ''.join(tokenize_set_name(card.set_code)) +
            ''.join(tokenize_power(card.power)) +
            ''.join(tokenize_toughness(card.toughness)) +
            ''.join(tokenize_type_line(card.type_line))
        )

    def test_tokenize_planeswalker(self):
        test_file = os.path.join(os.path.dirname(__file__), "test_data", "ob_nixilis_hate_twisted.json")
        with open(test_file, "r", encoding='utf-8') as f:
            data = json.load(f)
        card = Card.from_json(None, data)
        tokens = card.generate_tokens(["name", "mana_cost", "oracle_text", "release_year", "rarity", "set", "power", "toughness", "type_line", "loyalty"])
        self.assertEqual(
            ''.join(tokens),
            ''.join(tokenize_name(card.name)) +
            ''.join(tokenize_mana_cost(card.mana_cost)) +
            ''.join(tokenize_oracle_text(card.oracle_text, card_name=card.name, type_line=card.type_line)) +
            ''.join(tokenize_release_year(card.release_year)) +
            ''.join(tokenize_rarity(card.rarity)) +
            ''.join(tokenize_set_name(card.set_code)) +
            ''.join(tokenize_power(card.power)) +
            ''.join(tokenize_toughness(card.toughness)) +
            ''.join(tokenize_type_line(card.type_line)) +
            ''.join(tokenize_loyalty(card.loyalty))
        )

    def test_tokenize_token_creator(self):
        test_file = os.path.join(os.path.dirname(__file__), "test_data", "sprout_swarm.json")
        with open(test_file, "r") as f:
            data = json.load(f)
        card = Card.from_json(None, data)
        tokens = card.generate_tokens(["name", "mana_cost", "oracle_text", "release_year", "rarity", "set", "power", "toughness", "type_line", "loyalty"])
        self.assertEqual(
            ''.join(tokens),
            ''.join(tokenize_name(card.name)) +
            ''.join(tokenize_mana_cost(card.mana_cost)) +
            ''.join(tokenize_oracle_text(card.oracle_text, card_name=card.name, type_line=card.type_line)) +
            ''.join(tokenize_release_year(card.release_year)) +
            ''.join(tokenize_rarity(card.rarity)) +
            ''.join(tokenize_set_name(card.set_code)) +
            ''.join(tokenize_power(card.power)) +
            ''.join(tokenize_toughness(card.toughness)) +
            ''.join(tokenize_type_line(card.type_line)) +
            ''.join(tokenize_loyalty(card.loyalty))
        )
    
    def tokenize_nonexistent_field(self):
        test_file = os.path.join(os.path.dirname(__file__), "test_data", "grizzly_bear.json")
        with open(test_file, "r") as f:
            data = json.load(f)
        card = Card.from_json(None, data)
        with self.assertRaises(ValueError):
            card.generate_tokens(["nonexistent_field"])