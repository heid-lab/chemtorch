"""
Unit tests for token dataset operations.

This module tests individual functions in token_dataset_operations.py to ensure
they work correctly with mock datasets and representations.
"""

import pytest
import tempfile
import os
from typing import List, Dict

from chemtorch.components.dataset.token_dataset_operations import (
    get_vocab_size,
    extend_vocab,
    save_vocab,
    extend_vocab_from_strings,
    extend_vocab_from_data,
    extend_vocab_with_extractor,
)


class MockTokenRepresentation:
    """Mock token representation for testing."""
    
    def __init__(self, initial_vocab: Dict[str, int]):
        self._word2id = initial_vocab.copy()
        self._id2word = {v: k for k, v in initial_vocab.items()}
    
    @property
    def word2id(self) -> Dict[str, int]:
        return self._word2id
    
    @property
    def id2word(self) -> Dict[int, str]:
        return self._id2word
    
    @property
    def vocab_size(self) -> int:
        return len(self._word2id)
    
    def tokenize(self, input_str: str) -> List[str]:
        """Simple tokenization - split on spaces."""
        return input_str.split()
    
    def extend_vocab(self, new_tokens: List[str]) -> None:
        """Extend vocabulary with new tokens."""
        current_max_id = max(self._word2id.values()) if self._word2id else -1
        
        for token in new_tokens:
            if token not in self._word2id:
                current_max_id += 1
                self._word2id[token] = current_max_id
                self._id2word[current_max_id] = token
    
    def save_vocab(self, vocab_path: str) -> None:
        """Save vocabulary to file."""
        with open(vocab_path, "w", encoding="utf-8") as writer:
            for token, token_id in sorted(self._word2id.items(), key=lambda x: x[1]):
                writer.write(f"{token}\n")


class MockDataset:
    """Mock dataset for testing."""
    
    def __init__(self, representation, dataframe=None):
        self.representation = representation
        self.dataframe = dataframe


@pytest.fixture
def mock_representation():
    """Create a mock token representation with initial vocabulary."""
    initial_vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "C": 2,
        "N": 3,
        "O": 4,
    }
    return MockTokenRepresentation(initial_vocab)


@pytest.fixture
def mock_dataset(mock_representation):
    """Create a mock dataset with the mock representation."""
    return MockDataset(mock_representation)


class TestGetVocabSize:
    """Test cases for get_vocab_size function."""
    
    def test_get_vocab_size_basic(self, mock_dataset):
        """Test basic vocabulary size retrieval."""
        size = get_vocab_size(mock_dataset)
        assert size == 5  # Initial vocabulary has 5 tokens
    
    def test_get_vocab_size_empty_vocab(self):
        """Test vocab size with empty vocabulary."""
        empty_repr = MockTokenRepresentation({})
        dataset = MockDataset(empty_repr)
        
        size = get_vocab_size(dataset)
        assert size == 0


class TestExtendVocab:
    """Test cases for extend_vocab function."""
    
    def test_extend_vocab_new_tokens(self, mock_dataset):
        """Test extending vocabulary with new tokens."""
        initial_size = mock_dataset.representation.vocab_size
        new_tokens = ["H", "S", "P"]
        
        extend_vocab(mock_dataset, new_tokens)
        
        assert mock_dataset.representation.vocab_size == initial_size + 3
        assert "H" in mock_dataset.representation.word2id
        assert "S" in mock_dataset.representation.word2id
        assert "P" in mock_dataset.representation.word2id
    
    def test_extend_vocab_mixed_tokens(self, mock_dataset):
        """Test extending with mix of new and existing tokens, and empty list."""
        initial_size = mock_dataset.representation.vocab_size
        
        # Test with existing tokens only - should not change size
        extend_vocab(mock_dataset, ["C", "N"])
        assert mock_dataset.representation.vocab_size == initial_size
        
        # Test with mixed tokens - should add only new ones
        mixed_tokens = ["C", "H", "N", "S"]  # C and N exist, H and S are new
        extend_vocab(mock_dataset, mixed_tokens)
        assert mock_dataset.representation.vocab_size == initial_size + 2
        assert "H" in mock_dataset.representation.word2id
        assert "S" in mock_dataset.representation.word2id
        
        # Test with empty list - should not change size
        current_size = mock_dataset.representation.vocab_size
        extend_vocab(mock_dataset, [])
        assert mock_dataset.representation.vocab_size == current_size


class TestSaveVocab:
    """Test cases for save_vocab function."""
    
    def test_save_vocab_creates_file(self, mock_dataset):
        """Test that save_vocab creates a file with vocabulary."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp_file:
            vocab_path = tmp_file.name
        
        try:
            save_vocab(mock_dataset, vocab_path)
            
            # Verify file was created and has content
            assert os.path.exists(vocab_path)
            with open(vocab_path, 'r') as f:
                lines = f.readlines()
            
            # Should have 5 lines (one for each token in initial vocab)
            assert len(lines) == 5
            
            # Check that tokens are in the file
            tokens_in_file = [line.strip() for line in lines]
            assert "[PAD]" in tokens_in_file
            assert "[UNK]" in tokens_in_file
            assert "C" in tokens_in_file
            
        finally:
            if os.path.exists(vocab_path):
                os.unlink(vocab_path)


class TestExtendVocabFromStrings:
    """Test cases for extend_vocab_from_strings function."""
    
    def test_extend_vocab_from_strings_comprehensive(self, mock_dataset):
        """Test extending vocabulary from strings with various scenarios."""
        initial_size = mock_dataset.representation.vocab_size
        
        # Test basic functionality
        strings = ["C C O", "N H", "S P"]
        extend_vocab_from_strings(mock_dataset, strings)
        assert mock_dataset.representation.vocab_size == initial_size + 3
        assert "H" in mock_dataset.representation.word2id
        assert "S" in mock_dataset.representation.word2id
        assert "P" in mock_dataset.representation.word2id
        
        # Test with duplicates - should not add duplicates
        current_size = mock_dataset.representation.vocab_size
        duplicate_strings = ["H S", "H P", "S P"]  # H, S, P already exist
        extend_vocab_from_strings(mock_dataset, duplicate_strings)
        assert mock_dataset.representation.vocab_size == current_size  # No new tokens
        
        # Test with empty list
        extend_vocab_from_strings(mock_dataset, [])
        assert mock_dataset.representation.vocab_size == current_size


class TestExtendVocabFromData:
    """Test cases for extend_vocab_from_data function."""
    
    def test_extend_vocab_from_data_basic(self, mock_representation):
        """Test extending vocabulary from DataFrame column."""
        import pandas as pd
        
        # Create a DataFrame with SMILES column
        df = pd.DataFrame({
            'smiles': ['C C O', 'N H', 'S P'],
            'label': [1, 2, 3]
        })
        
        dataset = MockDataset(mock_representation, dataframe=df)
        initial_size = dataset.representation.vocab_size
        
        extend_vocab_from_data(dataset, column_name='smiles')
        
        # Should add H, S, P
        assert dataset.representation.vocab_size == initial_size + 3
    
    def test_extend_vocab_from_data_error_handling(self, mock_dataset):
        """Test error handling scenarios."""
        import pandas as pd
        
        # Test missing column
        df = pd.DataFrame({'other_column': ['data']})
        mock_dataset.dataframe = df
        
        with pytest.raises(ValueError, match="Column 'smiles' not found"):
            extend_vocab_from_data(mock_dataset, column_name='smiles')
        
        # Test no dataframe attribute
        delattr(mock_dataset, 'dataframe')
        with pytest.raises(ValueError, match="Dataset must have a 'dataframe' attribute"):
            extend_vocab_from_data(mock_dataset)

    def test_extend_vocab_from_data_with_nulls_and_custom_column(self, mock_representation):
        """Test handling of null values and custom column names."""
        import pandas as pd
        import numpy as np
        
        # Test with null values
        df = pd.DataFrame({
            'smiles': ['C C O', None, 'N H', np.nan, 'S P'],
            'custom_smiles': ['H P', 'S Cl', None, 'Br', 'F'],
            'label': [1, 2, 3, 4, 5]
        })
        
        dataset = MockDataset(mock_representation, dataframe=df)
        initial_size = dataset.representation.vocab_size
        
        # Test with nulls (should ignore them)
        extend_vocab_from_data(dataset, column_name='smiles')
        assert dataset.representation.vocab_size == initial_size + 3  # H, S, P
        
        # Test with custom column name
        current_size = dataset.representation.vocab_size
        extend_vocab_from_data(dataset, column_name='custom_smiles')
        assert dataset.representation.vocab_size == current_size + 3  # Cl, Br, F are new


class TestExtendVocabFromDataWithExtractor:
    """Test cases for extend_vocab_with_extractor function."""
    
    def test_extend_vocab_with_extractor_basic(self, mock_representation):
        """Test flexible vocabulary extension with custom extractor."""
        import pandas as pd
        
        df = pd.DataFrame({
            'smiles': ['C C O', 'N H'],
            'other_smiles': ['S P', 'Cl Br'],
            'label': [1, 2]
        })
        
        dataset = MockDataset(mock_representation, dataframe=df)
        
        def custom_extractor(ds):
            """Extract from both smiles columns."""
            strings = []
            strings.extend(ds.dataframe['smiles'].dropna().tolist())
            strings.extend(ds.dataframe['other_smiles'].dropna().tolist())
            return strings
        
        initial_size = dataset.representation.vocab_size
        
        extend_vocab_with_extractor(dataset, custom_extractor)
        
        # Should add H, S, P, Cl, Br
        assert dataset.representation.vocab_size == initial_size + 5
    
    def test_extend_vocab_with_extractor_custom_logic(self, mock_representation):
        """Test with complex extraction logic and empty extractor."""
        import pandas as pd
        
        df = pd.DataFrame({
            'type': ['reactant', 'product', 'reactant'],
            'smiles': ['C C O', 'N H', 'S P'],
            'label': [1, 2, 3]
        })
        
        dataset = MockDataset(mock_representation, dataframe=df)
        
        def reactant_only_extractor(ds):
            """Extract only reactant SMILES."""
            reactant_df = ds.dataframe[ds.dataframe['type'] == 'reactant']
            return reactant_df['smiles'].tolist()
        
        initial_size = dataset.representation.vocab_size
        
        extend_vocab_with_extractor(dataset, reactant_only_extractor)
        
        # Should add S, P (from "C C O" and "S P" - reactants only)
        assert dataset.representation.vocab_size == initial_size + 2
        
        # Test empty extractor
        def empty_extractor(ds):
            return []
        
        current_size = dataset.representation.vocab_size
        extend_vocab_with_extractor(dataset, empty_extractor)
        assert dataset.representation.vocab_size == current_size
