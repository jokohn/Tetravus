

class Card:
    def __init__(self, name, oracle_text, mana_cost, type_line, release_year, rarity, set_name):
        self.name = name
        self.oracle_text = oracle_text
        self.mana_cost = mana_cost
        self.type_line = type_line
        self.release_year = release_year
        self.rarity = rarity
        self.set_code = set_name

    def from_json(self, json_data):
        return Card(
            json_data["name"],
            json_data["oracle_text"],
            json_data["mana_cost"],
            json_data["type_line"],
            json_data["released_at"].split("-")[0],
            json_data["rarity"],
            json_data["set"]
        )
    
    def to_json(self):
        return {
            "name": self.name,
            "oracle_text": self.oracle_text,
            "mana_cost": self.mana_cost,
            "type_line": self.type_line,
            "release_year": self.release_year,
            "rarity": self.rarity,
            "set": self.set_code
        }