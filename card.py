from tokenizers.tokenize_oracle_text import tokenize_oracle_text
from tokenizers.tokenize_mana_cost import tokenize_mana_cost
from tokenizers.tokenize_simple_card_fields import tokenize_power, tokenize_toughness, tokenize_release_year, \
    tokenize_rarity, tokenize_set_name, tokenize_loyalty
from tokenizers.tokenize_name import tokenize_name
from tokenizers.tokenize_type_line import tokenize_type_line

class Card:
    def __init__(self, name, oracle_text, mana_cost, type_line, release_year, rarity, set_name, power=None, toughness=None, loyalty=None, all_parts=None, related_card_names=None):
        self.name = name
        self.oracle_text = oracle_text
        self.mana_cost = mana_cost
        self.type_line = type_line
        self.release_year = release_year
        self.rarity = rarity
        self.set_code = set_name
        self.power = power
        self.toughness = toughness
        self.loyalty = loyalty
        self.related_card_names = []
        if related_card_names is not None:
            self.related_card_names = related_card_names
        else:
            if all_parts is not None:
                for part in all_parts:
                    if not ('Legendary' not in part['type_line'] and part['component'] == 'token'):
                        self.related_card_names.append(part["name"])
    def from_json(self, json_data):
        try:
            return Card(
                json_data["name"],
                json_data.get("oracle_text"),
                json_data["mana_cost"],
                json_data["type_line"],
                json_data["released_at"].split("-")[0],
                json_data["rarity"],
                json_data["set"],
                json_data.get("power", None),
                json_data.get("toughness", None),
                json_data.get("loyalty", None),
                json_data.get("all_parts", None),
                json_data.get("related_card_names", None)
            )
        except KeyError as e:
            print(json_data)
            raise ValueError(f"Missing key: {e}")
    def to_json(self):
        card_json = {
            "name": self.name,
            "oracle_text": self.oracle_text,
            "mana_cost": self.mana_cost,
            "type_line": self.type_line,
            "release_year": self.release_year,
            "rarity": self.rarity,
            "set": self.set_code,
            "related_card_names": self.related_card_names,
        }
        if self.power:
            card_json["power"] = self.power
        if self.toughness:
            card_json["toughness"] = self.toughness
        if self.loyalty:
            card_json["loyalty"] = self.loyalty
        return card_json

    def generate_tokens(self, fields):
        tokens = []
        for field in fields:
            if field == 'name':
                tokens.extend(tokenize_name(self.name))
            elif field == 'oracle_text':
                tokens.extend(tokenize_oracle_text(self.oracle_text, self.name, self.type_line, self.related_card_names))
            elif field == 'mana_cost':
                tokens.extend(tokenize_mana_cost(self.mana_cost))
            elif field == 'type_line':
                tokens.extend(tokenize_type_line(self.type_line))
            elif field == 'release_year':
                tokens.extend(tokenize_release_year(self.release_year))
            elif field == 'rarity':
                tokens.extend(tokenize_rarity(self.rarity))
            elif field == 'set':
                tokens.extend(tokenize_set_name(self.set_code))
            elif field == 'power':
                if self.power is None:
                    continue
                tokens.extend(tokenize_power(self.power))
            elif field == 'toughness':
                if self.toughness is None:
                    continue
                tokens.extend(tokenize_toughness(self.toughness))
            elif field == 'loyalty':
                if self.loyalty is None:
                    continue
                tokens.extend(tokenize_loyalty(self.loyalty))
            else:
                raise ValueError(f"Invalid field: {field}")
        return tokens

    def is_complete(self):
        missing_fields = []
        if self.name is None:
            missing_fields.append('name')
        if self.oracle_text is None:
            missing_fields.append('oracle_text')
        if self.mana_cost is None:
            missing_fields.append('mana_cost')
        if self.type_line is None:
            missing_fields.append('type_line')
        if self.release_year is None:
            missing_fields.append('release_year')
        if self.rarity is None:
            missing_fields.append('rarity')
        if self.set_code is None:
            missing_fields.append('set')
        type_line = self.type_line or ""
        if 'Creature' in type_line or 'Vehicle' in type_line or 'Spacecraft' in type_line:
            if self.power is None:
                missing_fields.append('power')
            if self.toughness is None:
                missing_fields.append('toughness')
        if 'Planeswalker' in type_line:
            if self.loyalty is None:
                missing_fields.append('loyalty')
        return len(missing_fields) == 0, missing_fields

