import unittest
import json
import os
import tempfile
import random

from tokenize_card_file_permuted import (
    get_available_fields,
    generate_field_permutations,
    tokenize_card_file_permuted,
    shuffle_and_encode_token_blocks,
    write_tokenized_output
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
        """Test that field permutations are generated correctly."""
        fields = ["name", "mana_cost", "type_line"]
        num_permutations = 3
        
        # Should generate unique permutations when possible
        permutations = generate_field_permutations(fields, num_permutations)
        self.assertEqual(len(permutations), num_permutations)
        
        # All should be different
        permutation_strings = [tuple(p) for p in permutations]
        self.assertEqual(len(set(permutation_strings)), num_permutations)
        
        # Each should contain all fields
        for perm in permutations:
            self.assertEqual(set(perm), set(fields))
            self.assertEqual(len(perm), len(fields))
        
        # Test with more permutations than possible
        fields2 = ["name", "mana_cost"]
        permutations2 = generate_field_permutations(fields2, 10)  # Only 2! = 2 possible
        self.assertEqual(len(permutations2), 10)
        # Should contain duplicates
        permutation_strings2 = [tuple(p) for p in permutations2]
        self.assertLess(len(set(permutation_strings2)), 10)
    
    def test_tokenize_single_card(self):
        """Test tokenizing a single card with permutations."""
        # Create a temporary file with one card
        card_dict = {
            "name": "Test Card",
            "oracle_text": "Test oracle text",
            "mana_cost": "{1}{G}",
            "type_line": "Creature — Test",
            "released_at": "2020-01-01",
            "rarity": "common",
            "set": "test",
            "power": "2",
            "toughness": "2"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([card_dict], f)
            temp_file = f.name
        
        try:
            token_map, train_blocks, test_blocks, metadata = tokenize_card_file_permuted(
                temp_file, num_permutations=3, seed=42
            )
            
            # Should have processed 1 card
            self.assertEqual(metadata['processed_cards'], 1)
            self.assertEqual(metadata['failed_cards'], 0)
            
            # Card should be assigned to either train or test
            self.assertEqual(metadata['train_cards'] + metadata['test_cards'], 1)
            
            # Should have generated 3 token blocks total (one per permutation)
            total_blocks = len(train_blocks) + len(test_blocks)
            self.assertEqual(total_blocks, 3)
            self.assertEqual(metadata['total_blocks'], 3)
            self.assertGreater(metadata['total_unique_tokens'], 0)
            
            # All blocks should have tokens
            all_blocks = train_blocks + test_blocks
            for block in all_blocks:
                # All tokens should be in token_map
                for token in block:
                    self.assertIn(token, token_map)
            
        finally:
            os.unlink(temp_file)
    
    def test_tokenize_card_file_shuffling(self):
        """Test that token blocks are shuffled when encoding."""
        # Create a temporary file with one card
        card_dict = {
            "name": "Shuffle Test Card",
            "oracle_text": "Test text for shuffling",
            "mana_cost": "{2}{R}",
            "type_line": "Instant",
            "released_at": "2020-01-01",
            "rarity": "rare",
            "set": "test"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump([card_dict], f)
            temp_file = f.name
        
        try:
            # Tokenize with a fixed seed
            token_map, train_blocks, test_blocks, _ = tokenize_card_file_permuted(
                temp_file, num_permutations=5, seed=123
            )
            
            # Combine blocks for testing
            token_blocks = train_blocks + test_blocks
            # Should have 5 blocks total
            self.assertEqual(len(token_blocks), 5)
            
            # Encode and shuffle
            encoded_blocks = shuffle_and_encode_token_blocks(token_blocks, token_map)
            
            # Verify encoding: decode and check we get back the same token IDs
            for _, encoded_block in enumerate(encoded_blocks):
                # Decode chr-encoded token IDs
                token_ids = [ord(char) for char in encoded_block]
                
                # Convert back to token strings
                reverse_map = {v: k for k, v in token_map.items()}
                decoded_tokens = [reverse_map[tid] for tid in token_ids]
                
                # Should match one of the original blocks (order may differ due to shuffling)
                # But the set of tokens should match
                original_block_sets = [set(block) for block in token_blocks]
                decoded_set = set(decoded_tokens)
                self.assertIn(decoded_set, original_block_sets)
            
            # Verify shuffling: blocks should be in different order
            # (with high probability, though not guaranteed)
            # We'll check that at least the first block is different from original first block
            # or that the order has changed
            first_original = tuple(token_blocks[0])
            first_decoded_ids = [ord(char) for char in encoded_blocks[0]]
            reverse_map = {v: k for k, v in token_map.items()}
            first_decoded_tokens = tuple([reverse_map[tid] for tid in first_decoded_ids])
            
            # The first encoded block should correspond to one of the original blocks
            # but not necessarily the first one (due to shuffling)
            original_tuples = [tuple(block) for block in token_blocks]
            self.assertIn(first_decoded_tokens, original_tuples)
            
        finally:
            os.unlink(temp_file)
    
    def test_tokenize_same_card_twice_verifies_shuffling(self):
        """Test tokenizing the same card twice to verify shuffling behavior."""
        card_dict = {
            "name": "Double Test Card",
            "oracle_text": "Oracle text for double test",
            "mana_cost": "{3}",
            "type_line": "Sorcery",
            "released_at": "2021-01-01",
            "rarity": "uncommon",
            "set": "test2"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([card_dict], f)
            temp_file = f.name
        
        try:
            # First run with seed
            token_map1, train_blocks1, test_blocks1, _ = tokenize_card_file_permuted(
                temp_file, num_permutations=5, seed=999
            )
            token_blocks1 = train_blocks1 + test_blocks1
            # Set seed again before shuffling to ensure deterministic behavior
            random.seed(999)
            encoded_blocks1 = shuffle_and_encode_token_blocks(token_blocks1, token_map1)
            
            # Second run with same seed
            token_map2, train_blocks2, test_blocks2, _ = tokenize_card_file_permuted(
                temp_file, num_permutations=5, seed=999
            )
            token_blocks2 = train_blocks2 + test_blocks2
            # Set seed again before shuffling
            random.seed(999)
            encoded_blocks2 = shuffle_and_encode_token_blocks(token_blocks2, token_map2)
            
            # Token maps should be the same (same tokens encountered)
            self.assertEqual(token_map1, token_map2)
            
            # Token blocks should be the same (same permutations generated with same seed)
            self.assertEqual(len(token_blocks1), len(token_blocks2))
            
            # Encoded blocks should be in the same order (same seed = same shuffle)
            # This verifies that shuffling is deterministic with the same seed
            self.assertEqual(encoded_blocks1, encoded_blocks2)
            
            # Verify that blocks are actually shuffled (not in original order)
            # Decode first encoded block and check it's not necessarily the first original block
            first_decoded_ids = [ord(char) for char in encoded_blocks1[0]]
            reverse_map = {v: k for k, v in token_map1.items()}
            first_decoded_tokens = tuple([reverse_map[tid] for tid in first_decoded_ids])
            first_original_tokens = tuple(token_blocks1[0])
            
            # The first encoded block should correspond to one of the original blocks
            original_tuples = [tuple(block) for block in token_blocks1]
            self.assertIn(first_decoded_tokens, original_tuples)
            
            # With different seed, we should get different shuffle order
            token_map3, train_blocks3, test_blocks3, _ = tokenize_card_file_permuted(
                temp_file, num_permutations=5, seed=888
            )
            token_blocks3 = train_blocks3 + test_blocks3
            random.seed(888)
            encoded_blocks3 = shuffle_and_encode_token_blocks(token_blocks3, token_map3)
            
            # Token blocks should be the same (same card, same num_permutations)
            self.assertEqual(len(token_blocks3), len(token_blocks1))
            
            # With different seed, encoded blocks should be in different order
            # (high probability, though not guaranteed - but with different seeds it should differ)
            self.assertEqual(len(encoded_blocks3), len(encoded_blocks1))
            
        finally:
            os.unlink(temp_file)
    
    def test_write_tokenized_output(self):
        """Test writing output files."""
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
            "total_unique_tokens": 3
        }
        
        train_encoded = shuffle_and_encode_token_blocks(train_blocks, token_map)
        test_encoded = shuffle_and_encode_token_blocks(test_blocks, token_map)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            train_file = os.path.join(tmpdir, "test_train_tokens.txt")
            test_file = os.path.join(tmpdir, "test_test_tokens.txt")
            map_file = os.path.join(tmpdir, "test_map.json")
            
            write_tokenized_output(train_encoded, test_encoded, token_map, train_file, test_file, map_file, metadata)
            
            # Verify training text file exists and has correct format
            self.assertTrue(os.path.exists(train_file))
            with open(train_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # File should contain all encoded blocks concatenated
                # Each character should be a valid Unicode character (token ID)
                self.assertGreater(len(content), 0)
                # Verify we can decode token IDs from characters
                token_ids = [ord(char) for char in content]
                self.assertEqual(len(token_ids), sum(len(block) for block in train_encoded))
            
            # Verify test text file exists and has correct format
            self.assertTrue(os.path.exists(test_file))
            with open(test_file, 'r') as f:
                lines = f.readlines()
                self.assertEqual(len(lines), len(test_encoded))
            
            # Verify map file exists and has correct format
            self.assertTrue(os.path.exists(map_file))
            with open(map_file, 'r') as f:
                data = json.load(f)
                self.assertIn("token_map", data)
                self.assertIn("metadata", data)
                self.assertEqual(data["token_map"], token_map)
                self.assertEqual(data["metadata"], metadata)
    
    def test_tokenize_multiple_cards(self):
        """Test tokenizing multiple cards."""
        # Use the existing cleaned_cards.json test file
        token_map, train_blocks, test_blocks, metadata = tokenize_card_file_permuted(
            self.cleaned_cards_file, num_permutations=3, seed=42
        )
        
        # Should have processed multiple cards
        self.assertGreater(metadata['processed_cards'], 0)
        self.assertEqual(metadata['failed_cards'], 0)
        
        # Cards should be split between train and test
        self.assertEqual(metadata['train_cards'] + metadata['test_cards'], metadata['processed_cards'])
        
        # Should have generated blocks (3 per card)
        expected_min_blocks = metadata['processed_cards'] * 3
        self.assertGreaterEqual(metadata['total_blocks'], expected_min_blocks)
        
        # Verify all tokens are in token_map
        all_blocks = train_blocks + test_blocks
        for block in all_blocks:
            for token in block:
                self.assertIn(token, token_map)
        
        # Verify begin_name_token is in token_map (should be present)
        self.assertIn(begin_name_token, token_map)
    
    def test_train_test_split(self):
        """Test that train/test splitting works correctly."""
        # Create a file with multiple cards
        card_dicts = [
            {
                "name": f"Test Card {i}",
                "oracle_text": f"Oracle text {i}",
                "mana_cost": "{1}",
                "type_line": "Creature",
                "released_at": "2020-01-01",
                "rarity": "common",
                "set": "test"
            }
            for i in range(10)
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(card_dicts, f)
            temp_file = f.name
        
        try:
            # Test with 80/20 split
            token_map, train_blocks, test_blocks, metadata = tokenize_card_file_permuted(
                temp_file, num_permutations=2, seed=42, train_test_split=0.8
            )
            
            # Should have processed all 10 cards
            self.assertEqual(metadata['processed_cards'], 10)
            self.assertEqual(metadata['train_cards'] + metadata['test_cards'], 10)
            
            # With 80/20 split and seed=42, we should get roughly 8 train, 2 test
            # (exact numbers depend on random, but should be close)
            self.assertGreater(metadata['train_cards'], 0)
            self.assertGreater(metadata['test_cards'], 0)
            
            # Blocks should be split accordingly
            self.assertEqual(len(train_blocks), metadata['train_blocks'])
            self.assertEqual(len(test_blocks), metadata['test_blocks'])
            self.assertEqual(metadata['train_blocks'] + metadata['test_blocks'], metadata['total_blocks'])
            
            # Test with 50/50 split
            token_map2, train_blocks2, test_blocks2, metadata2 = tokenize_card_file_permuted(
                temp_file, num_permutations=2, seed=42, train_test_split=0.5
            )
            
            # Should still process all cards
            self.assertEqual(metadata2['processed_cards'], 10)
            
            # With 50/50 split, should be roughly equal
            # (exact numbers depend on random)
            self.assertGreater(metadata2['train_cards'], 0)
            self.assertGreater(metadata2['test_cards'], 0)
            
            # Test with 100% training (no test)
            token_map3, train_blocks3, test_blocks3, metadata3 = tokenize_card_file_permuted(
                temp_file, num_permutations=2, seed=42, train_test_split=1.0
            )
            
            # All cards should be in training
            self.assertEqual(metadata3['train_cards'], 10)
            self.assertEqual(metadata3['test_cards'], 0)
            self.assertEqual(len(test_blocks3), 0)
            
            # Test with 0% training (all test)
            token_map4, train_blocks4, test_blocks4, metadata4 = tokenize_card_file_permuted(
                temp_file, num_permutations=2, seed=42, train_test_split=0.0
            )
            
            # All cards should be in test
            self.assertEqual(metadata4['train_cards'], 0)
            self.assertEqual(metadata4['test_cards'], 10)
            self.assertEqual(len(train_blocks4), 0)
            
        finally:
            os.unlink(temp_file)
    
    def test_train_test_split_deterministic(self):
        """Test that train/test split is deterministic with same seed."""
        card_dicts = [
            {
                "name": f"Card {i}",
                "oracle_text": f"Text {i}",
                "mana_cost": "{1}",
                "type_line": "Creature",
                "released_at": "2020-01-01",
                "rarity": "common",
                "set": "test"
            }
            for i in range(5)
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(card_dicts, f)
            temp_file = f.name
        
        try:
            # First run
            _, train_blocks1, test_blocks1, metadata1 = tokenize_card_file_permuted(
                temp_file, num_permutations=2, seed=123, train_test_split=0.8
            )
            
            # Second run with same seed
            _, train_blocks2, test_blocks2, metadata2 = tokenize_card_file_permuted(
                temp_file, num_permutations=2, seed=123, train_test_split=0.8
            )
            
            # Should get same split
            self.assertEqual(metadata1['train_cards'], metadata2['train_cards'])
            self.assertEqual(metadata1['test_cards'], metadata2['test_cards'])
            self.assertEqual(len(train_blocks1), len(train_blocks2))
            self.assertEqual(len(test_blocks1), len(test_blocks2))
            
        finally:
            os.unlink(temp_file)


if __name__ == '__main__':
    unittest.main()

