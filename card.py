from tokenizers.tokenize_oracle_text import tokenize_oracle_text
from tokenizers.tokenize_mana_cost import tokenize_mana_cost
from tokenizers.tokenize_simple_card_fields import tokenize_power, tokenize_toughness, tokenize_release_year, \
    tokenize_rarity, tokenize_set_name
from tokenizers.tokenize_name import tokenize_name
from tokenizers.tokenize_type_line import tokenize_type_line

class Card:
    def __init__(self, name, oracle_text, mana_cost, type_line, release_year, rarity, set_name, power=None, toughness=None):
        self.name = name
        self.oracle_text = oracle_text
        self.mana_cost = mana_cost
        self.type_line = type_line
        self.release_year = release_year
        self.rarity = rarity
        self.set_code = set_name
        self.power = power
        self.toughness = toughness

    def from_json(self, json_data):
        return Card(
            json_data["name"],
            json_data["oracle_text"],
            json_data["mana_cost"],
            json_data["type_line"],
            json_data["released_at"].split("-")[0],
            json_data["rarity"],
            json_data["set"],
            json_data.get("power", None),
            json_data.get("toughness", None)
        )
    
    def to_json(self):
        card_json = {
            "name": self.name,
            "oracle_text": self.oracle_text,
            "mana_cost": self.mana_cost,
            "type_line": self.type_line,
            "release_year": self.release_year,
            "rarity": self.rarity,
            "set": self.set_code,
        }
        if self.power:
            card_json["power"] = self.power
        if self.toughness:
            card_json["toughness"] = self.toughness
        return card_json

    def generate_tokens(self, fields):
        tokens = ''
        for field in fields:
            if field == 'name':
                tokens += ''.join(tokenize_name(self.name))
            elif field == 'oracle_text':
                tokens += ''.join(tokenize_oracle_text(self.oracle_text))
            elif field == 'mana_cost':
                tokens += ''.join(tokenize_mana_cost(self.mana_cost))
            elif field == 'type_line':
                tokens += ''.join(tokenize_type_line(self.type_line))
            elif field == 'release_year':
                tokens += ''.join(tokenize_release_year(self.release_year))
            elif field == 'rarity':
                tokens += ''.join(tokenize_rarity(self.rarity))
            elif field == 'set':
                tokens += ''.join(tokenize_set_name(self.set_code))
            elif field == 'power':
                tokens += ''.join(tokenize_power(self.power))
            elif field == 'toughness':
                tokens += ''.join(tokenize_toughness(self.toughness))
            else:
                raise ValueError(f"Invalid field: {field}")
        return tokens