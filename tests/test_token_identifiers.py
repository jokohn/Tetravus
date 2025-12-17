import unittest

from tokenizers.oracle_text_helper_functions.oracle_text_type_identifiers import is_stat_change_token, \
    is_stat_definition_token, is_planeswalker_loyalty_ability_token, is_planeswalker_loyaly_ability_colon_token, \
    is_mana_cost_token, is_planeswalker_loyalty_ability_token

class TestTokenIdentifiers(unittest.TestCase):
    def test_is_stat_change_token(self):
        self.assertTrue(is_stat_change_token('+1/+1'))
        self.assertTrue(is_stat_change_token('−1/−1'))
        self.assertTrue(is_stat_change_token('+X/−X'))
        self.assertTrue(is_stat_change_token('−X/−X'))
        self.assertTrue(is_stat_definition_token('1/1'))
        self.assertTrue(is_stat_definition_token('X/X'))
        self.assertTrue(is_stat_definition_token('1/X'))
        self.assertTrue(is_stat_definition_token('X/1'))

    def test_is_mana_cost_token(self):
        self.assertTrue(is_mana_cost_token('{1}'))
        self.assertTrue(is_mana_cost_token('{1}{G}'))
        self.assertTrue(is_mana_cost_token('{W/U}'))
        self.assertTrue(is_mana_cost_token('{W/U}{W/U}'))
        self.assertTrue(is_mana_cost_token('{1}{1}'))
        self.assertTrue(is_mana_cost_token('{10}'))
        self.assertTrue(is_mana_cost_token('{1}{W/U}'))
        self.assertTrue(is_mana_cost_token('{10}{W/U}{B}'))
        