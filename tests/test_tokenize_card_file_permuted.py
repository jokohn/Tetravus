import unittest
import json
import os
import tempfile
import random

from tokenize_card_file_permuted import (
    get_available_fields,
    generate_field_permutations,
    tokenize_card_file_permuted,
    tokenize_card_file_fim,
    shuffle_and_encode_token_blocks,
    write_tokenized_output,
)
from card import Card
from special_tokens import begin_name_token


class TestTokenizeCardFilePermuted(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")
        self.cleaned_cards_file = os.path.join(self.test_data_dir, "cleaned_cards.json")
    
    def test_get_available_fields(self):
        """Test that get_available_fields correctly identifies available fields."""
        # Card with all optional fields
        card_dict = {
            "name": "Test Card",
            "oracle_text": "Test text",
            "mana_cost": "{1}",
            "type_line": "Creature",
            "released_at": "2020-01-01",
            "rarity": "common",
            "set": "test",
            "power": "2",
            "toughness": "3",
            "loyalty": "4"
        }
        card = Card.from_json(None, card_dict)
        fields = get_available_fields(card)
        
        # Should include all fields
        self.assertIn("name", fields)
        self.assertIn("mana_cost", fields)
        self.assertIn("type_line", fields)
        self.assertIn("release_year", fields)
        self.assertIn("rarity", fields)
        self.assertIn("set", fields)
        self.assertIn("oracle_text", fields)
        self.assertIn("power", fields)
        self.assertIn("toughness", fields)
        self.assertIn("loyalty", fields)
        
        # Card without optional fields
        card_dict2 = {
            "name": "Test Card 2",
            "oracle_text": "",
            "mana_cost": "{2}",
            "type_line": "Instant",
            "released_at": "2020-01-01",
            "rarity": "uncommon",
            "set": "test2"
        }
        card2 = Card.from_json(None, card_dict2)
        fields2 = get_available_fields(card2)
        
        # Should include oracle_text (always included now)
        self.assertIn("oracle_text", fields2)
        # Should not include power, toughness, loyalty
        self.assertNotIn("power", fields2)
        self.assertNotIn("toughness", fields2)
        self.assertNotIn("loyalty", fields2)
    
    def test_generate_field_permutations(self):
        """Test that field permutations are generated correctly (legacy; used by permuted path)."""
        fields = ["name", "mana_cost", "type_line"]
        num_permutations = 3
        permutations = generate_field_permutations(fields, num_permutations)
        self.assertEqual(len(permutations), num_permutations)
        permutation_strings = [tuple(p) for p in permutations]
        self.assertEqual(len(set(permutation_strings)), num_permutations)
        for perm in permutations:
            self.assertEqual(set(perm), set(fields))
            self.assertEqual(len(perm), len(fields))

    def test_tokenize_single_card_fim(self):
        """Test tokenizing a single card with FIM (one block per card, standard or FIM)."""
        card_dict = {
            "name": "Test Card",
            "oracle_text": "Test oracle text",
            "mana_cost": "{1}{G}",
            "type_line": "Creature — Test",
            "released_at": "2020-01-01",
            "rarity": "common",
            "set": "test",
            "power": "2",
            "toughness": "2",
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([card_dict], f)
            temp_file = f.name
        try:
            token_map, train_blocks, test_blocks, metadata = tokenize_card_file_fim(
                temp_file, seed=42, blocks_per_card=1
            )
            self.assertEqual(metadata["processed_cards"], 1)
            self.assertEqual(metadata["failed_cards"], 0)
            self.assertEqual(metadata["train_cards"] + metadata["test_cards"], 1)
            total_blocks = len(train_blocks) + len(test_blocks)
            self.assertEqual(total_blocks, 1)
            self.assertEqual(metadata["total_blocks"], 1)
            self.assertGreater(metadata["total_unique_tokens"], 0)
            all_blocks = train_blocks + test_blocks
            for block in all_blocks:
                for token in block:
                    self.assertIn(token, token_map)
        finally:
            os.unlink(temp_file)

    def test_tokenize_multiple_blocks_per_card(self):
        """Test that each card generates blocks_per_card blocks (half standard, half FIM)."""
        card_dict = {
            "name": "Multi Block Card",
            "oracle_text": "Oracle text",
            "mana_cost": "{1}{W}",
            "type_line": "Creature",
            "released_at": "2020-01-01",
            "rarity": "common",
            "set": "test",
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([card_dict], f)
            temp_file = f.name
        try:
            token_map, train_blocks, test_blocks, metadata = tokenize_card_file_fim(
                temp_file, seed=42, blocks_per_card=10
            )
            self.assertEqual(metadata["processed_cards"], 1)
            total_blocks = len(train_blocks) + len(test_blocks)
            self.assertEqual(total_blocks, 10)
            self.assertEqual(metadata["total_blocks"], 10)
        finally:
            os.unlink(temp_file)
    
    def test_tokenize_card_file_shuffling_fim(self):
        """Test that token blocks are shuffled when encoding (FIM pipeline)."""
        card_dict = {
            "name": "Shuffle Test Card",
            "oracle_text": "Test text for shuffling",
            "mana_cost": "{2}{R}",
            "type_line": "Instant",
            "released_at": "2020-01-01",
            "rarity": "rare",
            "set": "test",
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump([card_dict], f)
            temp_file = f.name
        try:
            token_map, train_blocks, test_blocks, _ = tokenize_card_file_fim(
                temp_file, seed=123, blocks_per_card=1
            )
            token_blocks = train_blocks + test_blocks
            self.assertEqual(len(token_blocks), 1)
            encoded_blocks = shuffle_and_encode_token_blocks(token_blocks, token_map)
            token_ids = [ord(char) for char in encoded_blocks[0]]
            reverse_map = {v: k for k, v in token_map.items()}
            decoded_tokens = [reverse_map[tid] for tid in token_ids]
            self.assertEqual(decoded_tokens, token_blocks[0])
        finally:
            os.unlink(temp_file)
    
    def test_tokenize_same_card_twice_fim_deterministic(self):
        """Test tokenizing the same card twice with FIM (deterministic with same seed)."""
        card_dict = {
            "name": "Double Test Card",
            "oracle_text": "Oracle text for double test",
            "mana_cost": "{3}",
            "type_line": "Sorcery",
            "released_at": "2021-01-01",
            "rarity": "uncommon",
            "set": "test2",
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([card_dict], f)
            temp_file = f.name
        try:
            token_map1, train_blocks1, test_blocks1, _ = tokenize_card_file_fim(
                temp_file, seed=999, blocks_per_card=1
            )
            token_blocks1 = train_blocks1 + test_blocks1
            random.seed(999)
            encoded_blocks1 = shuffle_and_encode_token_blocks(token_blocks1, token_map1)
            token_map2, train_blocks2, test_blocks2, _ = tokenize_card_file_fim(
                temp_file, seed=999, blocks_per_card=1
            )
            token_blocks2 = train_blocks2 + test_blocks2
            random.seed(999)
            encoded_blocks2 = shuffle_and_encode_token_blocks(token_blocks2, token_map2)
            self.assertEqual(token_map1, token_map2)
            self.assertEqual(len(token_blocks1), len(token_blocks2))
            self.assertEqual(encoded_blocks1, encoded_blocks2)
        finally:
            os.unlink(temp_file)
    
    def test_write_tokenized_output(self):
        """Test writing output files (no mask files)."""
        token_map = {"<token1>": 0, "<token2>": 1, "<token3>": 2}
        train_blocks = [["<token1>", "<token2>"]]
        test_blocks = [["<token2>", "<token3>"]]
        metadata = {
            "processed_cards": 1,
            "failed_cards": 0,
            "train_cards": 1,
            "test_cards": 0,
            "train_blocks": 1,
            "test_blocks": 1,
            "total_blocks": 2,
            "total_unique_tokens": 3,
        }
        train_encoded = shuffle_and_encode_token_blocks(train_blocks, token_map)
        test_encoded = shuffle_and_encode_token_blocks(test_blocks, token_map)
        with tempfile.TemporaryDirectory() as tmpdir:
            train_file = os.path.join(tmpdir, "test_train_tokens.txt")
            test_file = os.path.join(tmpdir, "test_test_tokens.txt")
            map_file = os.path.join(tmpdir, "test_map.json")
            write_tokenized_output(
                train_encoded, test_encoded, token_map,
                train_file, test_file, map_file, metadata,
            )
            self.assertTrue(os.path.exists(train_file))
            with open(train_file, "r", encoding="utf-8") as f:
                content = f.read()
                self.assertGreater(len(content), 0)
                token_ids = [ord(char) for char in content]
                self.assertEqual(len(token_ids), sum(len(block) for block in train_encoded))
            self.assertTrue(os.path.exists(test_file))
            with open(test_file, "r", encoding="utf-8") as f:
                content = f.read()
                self.assertEqual(len(content), sum(len(block) for block in test_encoded))
            self.assertTrue(os.path.exists(map_file))
            with open(map_file, "r") as f:
                data = json.load(f)
                self.assertIn("token_map", data)
                self.assertIn("metadata", data)
                self.assertEqual(data["token_map"], token_map)
                self.assertEqual(data["metadata"], metadata)
    
    def test_tokenize_multiple_cards_fim(self):
        """Test tokenizing multiple cards with FIM (one block per card)."""
        token_map, train_blocks, test_blocks, metadata = tokenize_card_file_fim(
            self.cleaned_cards_file, seed=42, blocks_per_card=1
        )
        self.assertGreater(metadata["processed_cards"], 0)
        self.assertEqual(metadata["failed_cards"], 0)
        self.assertEqual(
            metadata["train_cards"] + metadata["test_cards"],
            metadata["processed_cards"],
        )
        self.assertEqual(
            metadata["total_blocks"],
            metadata["train_blocks"] + metadata["test_blocks"],
        )
        self.assertEqual(metadata["total_blocks"], metadata["processed_cards"])
        all_blocks = train_blocks + test_blocks
        for block in all_blocks:
            for token in block:
                self.assertIn(token, token_map)
        self.assertIn(begin_name_token, token_map)
    
    def test_train_test_split_fim(self):
        """Test that train/test splitting works correctly with FIM."""
        card_dicts = [
            {
                "name": f"Test Card {i}",
                "oracle_text": f"Oracle text {i}",
                "mana_cost": "{1}",
                "type_line": "Creature",
                "released_at": "2020-01-01",
                "rarity": "common",
                "set": "test",
            }
            for i in range(10)
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(card_dicts, f)
            temp_file = f.name
        try:
            token_map, train_blocks, test_blocks, metadata = tokenize_card_file_fim(
                temp_file, seed=0, train_test_split=0.8, blocks_per_card=1
            )
            self.assertEqual(metadata["processed_cards"], 10)
            self.assertEqual(metadata["train_cards"] + metadata["test_cards"], 10)
            self.assertGreater(metadata["train_cards"], 0, "seed=0 should yield some training cards")
            self.assertGreater(metadata["test_cards"], 0, "seed=0 should yield some test cards")
            self.assertEqual(len(train_blocks), metadata["train_blocks"])
            self.assertEqual(len(test_blocks), metadata["test_blocks"])
            self.assertEqual(
                metadata["train_blocks"] + metadata["test_blocks"],
                metadata["total_blocks"],
            )
            _, _, _, metadata2 = tokenize_card_file_fim(
                temp_file, seed=0, train_test_split=0.5, blocks_per_card=1
            )
            self.assertEqual(metadata2["processed_cards"], 10)
            self.assertGreater(metadata2["train_cards"], 0)
            self.assertGreater(metadata2["test_cards"], 0)
            _, _, test_blocks3, metadata3 = tokenize_card_file_fim(
                temp_file, seed=0, train_test_split=1.0, blocks_per_card=1
            )
            self.assertEqual(metadata3["train_cards"], 10)
            self.assertEqual(metadata3["test_cards"], 0)
            self.assertEqual(len(test_blocks3), 0)
            _, train_blocks4, _, metadata4 = tokenize_card_file_fim(
                temp_file, seed=0, train_test_split=0.0, blocks_per_card=1
            )
            self.assertEqual(metadata4["train_cards"], 0)
            self.assertEqual(metadata4["test_cards"], 10)
            self.assertEqual(len(train_blocks4), 0)
        finally:
            os.unlink(temp_file)
    
    def test_train_test_split_deterministic_fim(self):
        """Test that train/test split is deterministic with same seed (FIM)."""
        card_dicts = [
            {
                "name": f"Card {i}",
                "oracle_text": f"Text {i}",
                "mana_cost": "{1}",
                "type_line": "Creature",
                "released_at": "2020-01-01",
                "rarity": "common",
                "set": "test",
            }
            for i in range(5)
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(card_dicts, f)
            temp_file = f.name
        try:
            _, train_blocks1, test_blocks1, metadata1 = tokenize_card_file_fim(
                temp_file, seed=123, train_test_split=0.8, blocks_per_card=1
            )
            _, train_blocks2, test_blocks2, metadata2 = tokenize_card_file_fim(
                temp_file, seed=123, train_test_split=0.8, blocks_per_card=1
            )
            self.assertEqual(metadata1["train_cards"], metadata2["train_cards"])
            self.assertEqual(metadata1["test_cards"], metadata2["test_cards"])
            self.assertEqual(len(train_blocks1), len(train_blocks2))
            self.assertEqual(len(test_blocks1), len(test_blocks2))
        finally:
            os.unlink(temp_file)


if __name__ == '__main__':
    unittest.main()

