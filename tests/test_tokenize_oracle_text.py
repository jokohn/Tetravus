import unittest
from re import sub
import json
import os

from card import Card
from tokenizers.tokenize_oracle_text import tokenize_oracle_text, detokenize_oracle_text
from token_stream import TokenStream
from tokenizers.oracle_text_helper_functions.preprocess_oracle_text import UnsupportedCharacterError
from special_tokens import begin_oracle_text_token, end_oracle_text_token, begin_oracle_text_mana_cost_token, \
    end_oracle_text_mana_cost_token, oracle_text_this_card_token, oracle_text_open_quote_token, \
        oracle_text_close_quote_token, begin_stats_definition_token, end_stats_definition_token, \
        begin_stats_change_token, end_stats_change_token, oracle_text_other_card_token

class TestTokenizeOracleText(unittest.TestCase):
    def test_tokenize_oracle_text_simple(self):
        oracle_text = "Draw three cards."
        tokens = tokenize_oracle_text(oracle_text)
        self.assertEqual(tokens, [
            begin_oracle_text_token,
            "<oracle_text_draw>",
            "<oracle_text_three>",
            "<oracle_text_cards>",
            "<oracle_text_.>",
            end_oracle_text_token
        ])
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_oracle_text(token_stream), oracle_text)

    def test_strip_reminder_text(self):
        oracle_text = "Create a blood token. (It's an artifact with \"{1}, {T}, Discard a card, Sacrifice this token: Draw a card.\")"
        tokens = tokenize_oracle_text(oracle_text)
        self.assertEqual(tokens, [
            begin_oracle_text_token,
            "<oracle_text_create>",
            "<oracle_text_a>",
            "<oracle_text_blood>",
            "<oracle_text_token>",
            "<oracle_text_.>",
            end_oracle_text_token
        ])
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_oracle_text(token_stream), 'Create a blood token.')

    def test_newline_characters(self):
        oracle_text = "Draw a card.\nGain 3 life. Win the game."
        tokens = tokenize_oracle_text(oracle_text)
        self.assertEqual(tokens, [
            begin_oracle_text_token,
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
            end_oracle_text_token
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
            begin_oracle_text_token,
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
            end_oracle_text_token
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
            begin_oracle_text_token,
            end_oracle_text_token
        ])
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_oracle_text(token_stream), card.oracle_text)

    def test_oracle_text_with_mana_cost(self):
        oracle_text = "Counter target spell unless its controller pays {1}."
        tokens = tokenize_oracle_text(oracle_text)
        self.assertEqual(tokens, [
            begin_oracle_text_token,
            "<oracle_text_counter>",
            "<oracle_text_target>",
            "<oracle_text_spell>",
            "<oracle_text_unless>",
            "<oracle_text_its>",
            "<oracle_text_controller>",
            "<oracle_text_pays>",
            begin_oracle_text_mana_cost_token,
            "<oracle_text_mana_cost_1>",
            end_oracle_text_mana_cost_token,
            "<oracle_text_.>",
            end_oracle_text_token
        ])
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_oracle_text(token_stream), oracle_text)

    def abandon_the_post(self):
        test_file = os.path.join(os.path.dirname(__file__), "test_data", "abandon_the_post.json")
        with open(test_file, "r") as f:
            data = json.load(f)
        card = Card.from_json(None, data)
        tokens = tokenize_oracle_text(card.oracle_text)
        self.assertEqual(tokens, [
            begin_oracle_text_token,
            '<oracle_text_up>',
            '<oracle_text_to>',
            '<oracle_text_two>',
            '<oracle_text_target>',
            '<oracle_text_creatures>',
            '<oracle_text_can\'t>',
            '<oracle_text_block>',
            '<oracle_text_this>',
            '<oracle_text_turn>',
            '<oracle_text_.>',
            '<oracle_text_\n>',
            '<oracle_text_Flashback>',
            begin_oracle_text_mana_cost_token,
            '<oracle_text_mana_cost_3>',
            '<oracle_text_mana_cost_r>',
            end_oracle_text_mana_cost_token,
            end_oracle_text_token
        ])
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_oracle_text(token_stream), 'Up to two target creatures can\'t block this turn.\nFlashback {3}{R}')

    def test_activated_ability(self):
        oracle_text = "{U}: Mill a card."
        tokens = tokenize_oracle_text(oracle_text)
        self.assertEqual(tokens, [
            begin_oracle_text_token,
            begin_oracle_text_mana_cost_token,
            '<oracle_text_mana_cost_u>',
            end_oracle_text_mana_cost_token,
            '<oracle_text_:>',
            '<oracle_text_mill>',
            '<oracle_text_a>',
            '<oracle_text_card>',
            '<oracle_text_.>',
            end_oracle_text_token
        ])
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_oracle_text(token_stream), oracle_text)

    def test_bonesplitter(self):
        pass

    def test_oracle_test_name_replacement(self):
        card_name = 'Pinger, Who Deals Damage'
        oracle_text = 'When Pinger, Who Deals Damage enters, deal 1 damage to any target.'
        tokens = tokenize_oracle_text(oracle_text, card_name=card_name)
        self.assertEqual(tokens, [
            begin_oracle_text_token,
            '<oracle_text_when>',
            oracle_text_this_card_token,
            '<oracle_text_enters>',
            '<oracle_text_,>',
            '<oracle_text_deal>',
            '<oracle_text_1>',
            '<oracle_text_damage>',
            '<oracle_text_to>',
            '<oracle_text_any>',
            '<oracle_text_target>',
            '<oracle_text_.>',
            end_oracle_text_token
        ])
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_oracle_text(token_stream, card_name=card_name), oracle_text)

    def test_lightning_bolt(self):
        test_file = os.path.join(os.path.dirname(__file__), "test_data", "lightning_bolt.json")
        with open(test_file, "r") as f:
            data = json.load(f)
        card = Card.from_json(None, data)
        tokens = tokenize_oracle_text(card.oracle_text, card_name=card.name)
        self.assertEqual(tokens, [
            begin_oracle_text_token,
            oracle_text_this_card_token,
            '<oracle_text_deals>',
            '<oracle_text_3>',
            '<oracle_text_damage>',
            '<oracle_text_to>',
            '<oracle_text_any>',
            '<oracle_text_target>',
            '<oracle_text_.>',
            end_oracle_text_token
        ])
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_oracle_text(token_stream, card_name=card.name), card.oracle_text)

        card.generate_tokens(["oracle_text"]) == tokens

    def test_legendary_card_name_replacement(self):
        card_name = 'Pinger, Who Deals Damage'
        type_line = 'Legendary Creature'
        oracle_text = 'When Pinger enters, deal 1 damage to any target.'
        tokens = tokenize_oracle_text(oracle_text, card_name=card_name, type_line=type_line)
        self.assertEqual(tokens, [
            begin_oracle_text_token,
            '<oracle_text_when>',
            oracle_text_this_card_token,
            '<oracle_text_enters>',
            '<oracle_text_,>',
            '<oracle_text_deal>',
            '<oracle_text_1>',
            '<oracle_text_damage>',
            '<oracle_text_to>',
            '<oracle_text_any>',
            '<oracle_text_target>',
            '<oracle_text_.>',
            end_oracle_text_token
        ])
        token_stream = TokenStream(tokens)
        self.assertEqual(
            detokenize_oracle_text(token_stream, card_name=card_name),
            'When Pinger, Who Deals Damage enters, deal 1 damage to any target.'
        )

    def test_legendary_card_name_with_of_in_the_name_replacement(self):
        card_name = 'Pinger of the Ass Kickers'
        type_line = 'Legendary Creature'
        oracle_text = 'When Pinger enters, deal 1 damage to any target.'
        tokens = tokenize_oracle_text(oracle_text, card_name=card_name, type_line=type_line)
        self.assertEqual(tokens, [
            begin_oracle_text_token,
            '<oracle_text_when>',
            oracle_text_this_card_token,
            '<oracle_text_enters>',
            '<oracle_text_,>',
            '<oracle_text_deal>',
            '<oracle_text_1>',
            '<oracle_text_damage>',
            '<oracle_text_to>',
            '<oracle_text_any>',
            '<oracle_text_target>',
            '<oracle_text_.>',
            end_oracle_text_token
        ])
        token_stream = TokenStream(tokens)
        self.assertEqual(
            detokenize_oracle_text(token_stream, card_name=card_name),
            'When Pinger of the Ass Kickers enters, deal 1 damage to any target.'
        )

    def test_agrus_kos(self):
        test_file = os.path.join(os.path.dirname(__file__), "test_data", "agrus_kos.json")
        with open(test_file, "r") as f:
            data = json.load(f)
        card = Card.from_json(None, data)
        tokens = card.generate_tokens(["oracle_text"])
        self.assertEqual(tokens[:11], [
            begin_oracle_text_token,
            '<oracle_text_double>',
            '<oracle_text_strike>',
            '<oracle_text_,>',
            '<oracle_text_vigilance>',
            '<oracle_text_\n>',
            '<oracle_text_whenever>',
            oracle_text_this_card_token,
            '<oracle_text_enters>',
            '<oracle_text_or>',
            '<oracle_text_attacks>',
        ])

    def test_quoted_text(self):
        oracle_text = 'Create an artifact token with "Creatures you control have trample."'
        tokens = tokenize_oracle_text(oracle_text)
        self.assertEqual(tokens, [
            begin_oracle_text_token,
            '<oracle_text_create>',
            '<oracle_text_an>',
            '<oracle_text_artifact>',
            '<oracle_text_token>',
            '<oracle_text_with>',
            oracle_text_open_quote_token,
            '<oracle_text_creatures>',
            '<oracle_text_you>',
            '<oracle_text_control>',
            '<oracle_text_have>',
            '<oracle_text_trample>',
            '<oracle_text_.>',
            oracle_text_close_quote_token,
            end_oracle_text_token
        ])
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_oracle_text(token_stream), oracle_text)


    def test_abundant_growth(self):
        test_file = os.path.join(os.path.dirname(__file__), "test_data", "abundant_growth.json")
        with open(test_file, "r") as f:
            data = json.load(f)
        card = Card.from_json(None, data)
        tokens = card.generate_tokens(["oracle_text"])
        self.assertEqual(tokens,
            [
                begin_oracle_text_token,
                '<oracle_text_enchant>',
                '<oracle_text_land>',
                '<oracle_text_\n>',
                '<oracle_text_when>',
                '<oracle_text_this>',
                '<oracle_text_aura>',
                '<oracle_text_enters>',
                '<oracle_text_,>',
                '<oracle_text_draw>',
                '<oracle_text_a>',
                '<oracle_text_card>',
                '<oracle_text_.>',
                '<oracle_text_\n>',
                '<oracle_text_enchanted>',
                '<oracle_text_land>',
                '<oracle_text_has>',
                oracle_text_open_quote_token,
                begin_oracle_text_mana_cost_token,
                '<oracle_text_mana_cost_t>',
                end_oracle_text_mana_cost_token,
                '<oracle_text_:>',
                '<oracle_text_add>',
                '<oracle_text_one>',
                '<oracle_text_mana>',
                '<oracle_text_of>',
                '<oracle_text_any>',
                '<oracle_text_color>',
                '<oracle_text_.>',
                oracle_text_close_quote_token,
                end_oracle_text_token
            ]
        )
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_oracle_text(token_stream), card.oracle_text.replace('Aura', 'aura'))

    def test_drought(self):
        test_file = os.path.join(os.path.dirname(__file__), "test_data", "drought.json")
        with open(test_file, "r") as f:
            data = json.load(f)
        card = Card.from_json(None, data)
        tokens = card.generate_tokens(["oracle_text"])
        self.assertEqual(tokens,
            [
                begin_oracle_text_token,
                '<oracle_text_at>',
                '<oracle_text_the>',
                '<oracle_text_beginning>',
                '<oracle_text_of>',
                '<oracle_text_your>',
                '<oracle_text_upkeep>',
                '<oracle_text_,>',
                '<oracle_text_sacrifice>',
                '<oracle_text_this>',
                '<oracle_text_enchantment>',
                '<oracle_text_unless>',
                '<oracle_text_you>',
                '<oracle_text_pay>',
                begin_oracle_text_mana_cost_token,
                '<oracle_text_mana_cost_w>',
                '<oracle_text_mana_cost_w>',
                end_oracle_text_mana_cost_token,
                '<oracle_text_.>',
                '<oracle_text_\n>',
                '<oracle_text_spells>',
                '<oracle_text_cost>',
                '<oracle_text_an>',
                '<oracle_text_additional>',
                oracle_text_open_quote_token,
                '<oracle_text_sacrifice>',
                '<oracle_text_a>',
                '<oracle_text_swamp>',
                oracle_text_close_quote_token,
                '<oracle_text_to>',
                '<oracle_text_cast>',
                '<oracle_text_for>',
                '<oracle_text_each>',
                '<oracle_text_black>',
                '<oracle_text_mana>',
                '<oracle_text_symbol>',
                '<oracle_text_in>',
                '<oracle_text_their>',
                '<oracle_text_mana>',
                '<oracle_text_costs>',
                '<oracle_text_.>',
                '<oracle_text_\n>',
                '<oracle_text_activated>',
                '<oracle_text_abilities>',
                '<oracle_text_cost>',
                '<oracle_text_an>',
                '<oracle_text_additional>',
                oracle_text_open_quote_token,
                '<oracle_text_sacrifice>',
                '<oracle_text_a>',
                '<oracle_text_swamp>',
                oracle_text_close_quote_token,
                '<oracle_text_to>',
                '<oracle_text_activate>',
                '<oracle_text_for>',
                '<oracle_text_each>',
                '<oracle_text_black>',
                '<oracle_text_mana>',
                '<oracle_text_symbol>',
                '<oracle_text_in>',
                '<oracle_text_their>',
                '<oracle_text_activation>',
                '<oracle_text_costs>',
                '<oracle_text_.>',
                end_oracle_text_token
            ]
        )
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_oracle_text(token_stream), card.oracle_text.replace('Swamp', 'swamp'))

    def test_stat_definition_token(self):
        token_string = 'Create a 1/1 Saporling token.'
        tokens = tokenize_oracle_text(token_string)
        self.assertEqual(tokens, [
            begin_oracle_text_token,
            '<oracle_text_create>',
                '<oracle_text_a>',
                begin_stats_definition_token,
                '<stats_definition_power_1>',
                '<stats_definition_toughness_1>',
                end_stats_definition_token,
            '<oracle_text_saporling>',
            '<oracle_text_token>',
            '<oracle_text_.>',
            end_oracle_text_token
        ])
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_oracle_text(token_stream), token_string.replace('Saporling', 'saporling'))

    def test_stat_change_token(self):
        token_string = 'Target creature gets +2/+2 until end of turn.'
        tokens = tokenize_oracle_text(token_string)
        self.assertEqual(tokens, [
            begin_oracle_text_token,
            '<oracle_text_target>',
            '<oracle_text_creature>',
            '<oracle_text_gets>',
            begin_stats_change_token,
            '<stats_change_power_sign_+>',
            '<stats_change_power_value_2>',
            '<stats_change_toughness_sign_+>',
            '<stats_change_toughness_value_2>',
            end_stats_change_token,
            '<oracle_text_until>',
            '<oracle_text_end>',
            '<oracle_text_of>',
            '<oracle_text_turn>',
            '<oracle_text_.>',
            end_oracle_text_token
        ])
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_oracle_text(token_stream), token_string)

    def test_giant_growth(self):
        test_file = os.path.join(os.path.dirname(__file__), "test_data", "giant_growth.json")
        with open(test_file, "r") as f:
            data = json.load(f)
        card = Card.from_json(None, data)
        tokens = card.generate_tokens(["oracle_text"])
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_oracle_text(token_stream), card.oracle_text)

    def test_related_card_names(self):
        oracle_text = 'Creatures named Other Guy have trample.'
        tokens = tokenize_oracle_text(oracle_text, card_name='Test Card', related_card_names=['Other Guy'])
        self.assertEqual(tokens, [
            begin_oracle_text_token,
            '<oracle_text_creatures>',
            '<oracle_text_named>',
            oracle_text_other_card_token,
            '<oracle_text_have>',
            '<oracle_text_trample>',
            '<oracle_text_.>',
            end_oracle_text_token
        ])
        token_stream = TokenStream(tokens)
        self.assertEqual(detokenize_oracle_text(token_stream), oracle_text.replace('Other Guy', 'ANOTHER_CARD_NAME'))

    def test_helm_of_kaldra(self):
        test_file = os.path.join(os.path.dirname(__file__), "test_data", "helm_of_kaldra.json")
        with open(test_file, "r") as f:
            data = json.load(f)
        card = Card.from_json(None, data)
        tokens = card.generate_tokens(["oracle_text"])
        self.assertEqual(tokens, [
            begin_oracle_text_token,
            '<oracle_text_equipped>',
            '<oracle_text_creature>',
            '<oracle_text_has>',
            '<oracle_text_first>',
            '<oracle_text_strike>',
            '<oracle_text_,>',
            '<oracle_text_trample>',
            '<oracle_text_,>',
            '<oracle_text_and>',
            '<oracle_text_haste>',
            '<oracle_text_.>',
            '<oracle_text_\n>',
            begin_oracle_text_mana_cost_token,
            '<oracle_text_mana_cost_1>',
            end_oracle_text_mana_cost_token,
            '<oracle_text_:>',
            '<oracle_text_if>',
            '<oracle_text_you>',
            '<oracle_text_control>',
            '<oracle_text_equipment>',
            '<oracle_text_named>',
            oracle_text_this_card_token,
            '<oracle_text_,>',
            oracle_text_other_card_token,
            '<oracle_text_,>',
            '<oracle_text_and>',
            oracle_text_other_card_token,
            '<oracle_text_,>',
            '<oracle_text_create>',
            '<oracle_text_kaldra>',
            '<oracle_text_,>',
            '<oracle_text_a>',
            '<oracle_text_legendary>',
            begin_stats_definition_token,
            '<stats_definition_power_4>',
            '<stats_definition_toughness_4>',
            end_stats_definition_token,
            '<oracle_text_colorless>',
            '<oracle_text_avatar>',
            '<oracle_text_creature>',
            '<oracle_text_token>',
            '<oracle_text_.>',
            '<oracle_text_attach>',
            '<oracle_text_those>',
            '<oracle_text_equipment>',
            '<oracle_text_to>',
            '<oracle_text_it>',
            '<oracle_text_.>',
            '<oracle_text_\n>',
            '<oracle_text_equip>',
            begin_oracle_text_mana_cost_token,
            '<oracle_text_mana_cost_2>',
            end_oracle_text_mana_cost_token,
            end_oracle_text_token
        ])
        token_stream = TokenStream(tokens)
        self.assertEqual(
            detokenize_oracle_text(token_stream, card.name).lower(),
            card.oracle_text.lower().replace('sword of kaldra', 'another_card_name').replace('shield of kaldra', 'another_card_name')
        )

    def test_unsupported_characters(self):
        oracle_text = "Ward—Pay 2 Life|"
        with self.assertRaises(UnsupportedCharacterError):
            tokenize_oracle_text(oracle_text)
