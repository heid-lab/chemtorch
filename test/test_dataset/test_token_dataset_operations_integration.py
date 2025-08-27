"""
Integration tests for token dataset operations with real ChemTorch components.

This module tests the token dataset operations with actual DatasetBase,
TokenRepresentationBase, and DataModule components to ensure they work
correctly in the real framework context.
"""

import pytest
import pandas as pd
import torch
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock

from chemtorch.components.dataset.dataset_base import DatasetBase
from chemtorch.components.dataset.token_dataset_operations import (
    extend_vocab_from_data,
    extend_vocab_from_strings,
    get_vocab_size,
    save_vocab,
)
from chemtorch.components.representation.token.token_representation_base import TokenRepresentationBase
from chemtorch.core.data_module import DataModule
from chemtorch.utils.types import DataSplit


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def tokenize(self, smiles: str) -> list[str]:
        """Simple tokenization - just split on spaces for testing."""
        return smiles.split()


@pytest.fixture
def sample_vocab_file():
    """Create a temporary vocabulary file for testing."""
    vocab_content = """[PAD]
[UNK]
C
N
O
c
n
o
(
)
=
-
+
#
"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp_file:
        tmp_file.write(vocab_content)
        vocab_path = tmp_file.name
    
    yield vocab_path
    
    # Cleanup
    if os.path.exists(vocab_path):
        os.unlink(vocab_path)


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    return MockTokenizer()


@pytest.fixture
def token_representation(sample_vocab_file, mock_tokenizer):
    """Create a TokenRepresentationBase for testing."""
    return TokenRepresentationBase(
        vocab_path=sample_vocab_file,
        tokenizer=mock_tokenizer,
        max_sentence_length=50,
        pad_token="[PAD]",
        unk_token="[UNK]",
    )


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'smiles': [
            'C C O',
            'N N ( C )',
            'c c c c c c',
            'C = C',
            'N # N',
        ],
        'label': [0, 1, 2, 3, 4]
    })


@pytest.fixture
def token_dataset(sample_dataframe, token_representation):
    """Create a token dataset for testing."""
    return DatasetBase(
        dataframe=sample_dataframe,
        representation=token_representation,
        precompute_all=True,
        subsample=None,
    )


class TestTokenDatasetOperationsIntegration:
    """Integration tests for token dataset operations."""
    
    def test_get_vocab_size_real_dataset(self, token_dataset):
        """Test getting vocabulary size from real dataset."""
        vocab_size = get_vocab_size(token_dataset)
        
        # Should match the initial vocabulary size (14 tokens from fixture)
        assert vocab_size == 14
        assert vocab_size == token_dataset.representation.vocab_size
    
    def test_extend_vocab_from_data_real_dataset(self, token_dataset):
        """Test extending vocabulary from real dataset data."""
        initial_size = get_vocab_size(token_dataset)
        
        # Add some new data with new tokens
        new_data = pd.DataFrame({
            'smiles': ['S S', 'P P', 'Br Cl', 'F I'],
            'label': [5, 6, 7, 8]
        })
        
        # Extend the dataset's dataframe
        extended_df = pd.concat([token_dataset.dataframe, new_data], ignore_index=True)
        token_dataset.dataframe = extended_df
        
        # Extend vocabulary from the new data
        extend_vocab_from_data(token_dataset, column_name='smiles')
        
        # Should add S, P, Br, Cl, F, I (6 new tokens)
        assert get_vocab_size(token_dataset) == initial_size + 6
        
        # Verify the new tokens are in vocabulary
        assert 'S' in token_dataset.representation.word2id
        assert 'P' in token_dataset.representation.word2id
        assert 'Br' in token_dataset.representation.word2id
        assert 'Cl' in token_dataset.representation.word2id
        assert 'F' in token_dataset.representation.word2id
        assert 'I' in token_dataset.representation.word2id
    
    def test_extend_vocab_from_strings_real_dataset(self, token_dataset):
        """Test extending vocabulary from string list."""
        initial_size = get_vocab_size(token_dataset)
        
        new_strings = ['H H', 'S = S', 'P P P']
        extend_vocab_from_strings(token_dataset, new_strings)

        # Should add H, S, P (3 new tokens, = already exists)
        assert get_vocab_size(token_dataset) == initial_size + 3
        
        # Verify tokens are added
        assert 'H' in token_dataset.representation.word2id
        assert 'S' in token_dataset.representation.word2id
        assert 'P' in token_dataset.representation.word2id
    
    def test_save_vocab_real_dataset(self, token_dataset):
        """Test saving vocabulary from real dataset."""
        # First extend vocabulary with some new tokens
        extend_vocab_from_strings(token_dataset, ['H', 'S', 'P'])
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp_file:
            vocab_path = tmp_file.name
        
        try:
            save_vocab(token_dataset, vocab_path)
            
            # Verify file exists and has correct content
            assert os.path.exists(vocab_path)
            
            with open(vocab_path, 'r') as f:
                saved_tokens = [line.strip() for line in f.readlines()]
            
            # Should have original + 3 new tokens
            assert len(saved_tokens) == 17
            
            # Check that new tokens are saved
            assert 'H' in saved_tokens
            assert 'S' in saved_tokens
            assert 'P' in saved_tokens
            
            # Check that original tokens are still there
            assert '[PAD]' in saved_tokens
            assert '[UNK]' in saved_tokens
            assert 'C' in saved_tokens
            
        finally:
            if os.path.exists(vocab_path):
                os.unlink(vocab_path)
    
    def test_dataset_construction_after_vocab_extension(self, token_dataset):
        """Test that dataset can construct representations after vocabulary extension."""
        # Extend vocabulary
        extend_vocab_from_strings(token_dataset, ['H', 'S'])
        
        # Test that we can construct representations with new tokens
        new_smiles = 'H S'
        representation = token_dataset.representation.construct(new_smiles)
        
        # Should be a tensor
        assert isinstance(representation, torch.Tensor)
        assert representation.shape[0] == 1  # Batch dimension
        assert representation.shape[1] == token_dataset.representation.max_sentence_length
        
        # Verify that H and S are properly mapped (not UNK)
        tokens = token_dataset.representation.tokenize(new_smiles)
        token_ids = [token_dataset.representation.word2id[token] for token in tokens]
        
        # Should not be UNK token ID
        unk_id = token_dataset.representation.word2id['[UNK]']
        assert all(tid != unk_id for tid in token_ids)
    
    def test_dataset_item_access_after_vocab_extension(self, token_dataset):
        """Test that dataset item access works after vocabulary extension."""
        # Get an item before extension
        item_before = token_dataset[0]
        
        # Extend vocabulary
        extend_vocab_from_strings(token_dataset, ['H', 'S'])
        
        # Get the same item after extension
        item_after = token_dataset[0]
        
        # Should be the same (vocabulary extension doesn't change existing data)
        if token_dataset.has_labels:
            data_before, label_before = item_before
            data_after, label_after = item_after
            assert torch.equal(data_before, data_after)
            assert torch.equal(label_before, label_after)
        else:
            assert torch.equal(item_before, item_after)
    
    def test_vocab_extension_with_existing_tokens(self, token_dataset):
        """Test that extending with existing tokens doesn't increase size."""
        initial_size = get_vocab_size(token_dataset)
        
        # Try to add tokens that already exist
        existing_tokens = ['C', 'N', 'O', '(', ')']
        extend_vocab_from_strings(token_dataset, existing_tokens)
        
        # No tokens should be added
        assert get_vocab_size(token_dataset) == initial_size
    
    def test_vocab_extension_mixed_new_and_existing(self, token_dataset):
        """Test extending with mix of new and existing tokens."""
        initial_size = get_vocab_size(token_dataset)
        
        # Mix of new and existing tokens
        mixed_tokens = ['C', 'H', 'N', 'S', 'O', 'P']  # C, N, O exist; H, S, P are new
        extend_vocab_from_strings(token_dataset, mixed_tokens)
        
        # Should add only the 3 new tokens
        assert get_vocab_size(token_dataset) == initial_size + 3
    
    def test_vocab_consistency_after_operations(self, token_dataset):
        """Test that word2id and id2word remain consistent after operations."""
        # Extend vocabulary
        extend_vocab_from_strings(token_dataset, ['H', 'S', 'P'])
        
        word2id = token_dataset.representation.word2id
        id2word = token_dataset.representation.id2word
        
        # Check bidirectional consistency
        for word, idx in word2id.items():
            assert id2word[idx] == word
        
        for idx, word in id2word.items():
            assert word2id[word] == idx
        
        # Check that all IDs are unique
        assert len(set(word2id.values())) == len(word2id)
        assert len(set(id2word.keys())) == len(id2word)


class TestDataModuleIntegration:
    """Integration tests with DataModule."""
    
    def test_dataset_operations_with_data_module(self, sample_vocab_file, mock_tokenizer, sample_dataframe):
        """Test dataset operations applied through DataModule."""
        
        def mock_data_pipeline():
            """Mock data pipeline returning a DataSplit."""
            return DataSplit(
                train=sample_dataframe.iloc[:3],
                val=sample_dataframe.iloc[3:4],
                test=sample_dataframe.iloc[4:5],
            )
        
        def dataset_factory(df):
            """Factory function to create datasets."""
            representation = TokenRepresentationBase(
                vocab_path=sample_vocab_file,
                tokenizer=mock_tokenizer,
                max_sentence_length=50,
                pad_token="[PAD]",
                unk_token="[UNK]",
            )
            return DatasetBase(
                dataframe=df,
                representation=representation,
                precompute_all=True,
            )
        
        def dataloader_factory(dataset, shuffle):
            """Mock dataloader factory."""
            from torch.utils.data import DataLoader
            return DataLoader(dataset, batch_size=2, shuffle=shuffle)
        
        # Create DataModule
        data_module = DataModule(
            data_pipeline=mock_data_pipeline,
            dataset_factory=dataset_factory,
            dataloader_factory=dataloader_factory,
        )
        
        # Check initial vocabulary sizes
        train_vocab_size = get_vocab_size(data_module.datasets['train'])
        val_vocab_size = get_vocab_size(data_module.datasets['val'])
        test_vocab_size = get_vocab_size(data_module.datasets['test'])
        
        assert train_vocab_size == val_vocab_size == test_vocab_size == 14
        
        # Create a dataset operation
        def extend_vocab_operation(dataset):
            extend_vocab_from_strings(dataset, ['H', 'S', 'P'])
        
        # Apply operation to all datasets
        data_module.apply_dataset_operation(extend_vocab_operation)
        
        # Check that all datasets have extended vocabulary
        assert get_vocab_size(data_module.datasets['train']) == 17
        assert get_vocab_size(data_module.datasets['val']) == 17
        assert get_vocab_size(data_module.datasets['test']) == 17
        
        # Verify that new tokens are in all datasets
        for dataset_key in ['train', 'val', 'test']:
            dataset = data_module.datasets[dataset_key]
            assert 'H' in dataset.representation.word2id
            assert 'S' in dataset.representation.word2id
            assert 'P' in dataset.representation.word2id
    

class TestErrorHandling:
    """Test error handling in integration scenarios."""
    
    def test_extend_vocab_from_data_missing_column(self, token_dataset):
        """Test error handling when column doesn't exist."""
        with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
            extend_vocab_from_data(token_dataset, column_name='nonexistent')
    
    def test_tokenization_error_handling(self, sample_vocab_file):
        """Test handling of tokenization errors."""
        
        class ErrorTokenizer:
            def tokenize(self, smiles: str):
                if 'error' in smiles:
                    raise ValueError("Tokenization failed")
                return smiles.split()
        
        representation = TokenRepresentationBase(
            vocab_path=sample_vocab_file,
            tokenizer=ErrorTokenizer(),
            max_sentence_length=50,
            pad_token="[PAD]",
            unk_token="[UNK]",
        )
        
        df = pd.DataFrame({
            'smiles': ['C C', 'error string', 'N N'],
            'label': [1, 2, 3]
        })
        
        # should throw a runtime error
        with pytest.raises(RuntimeError, match="Tokenization failed"):
            DatasetBase(
                dataframe=df,
                representation=representation,
                precompute_all=True,
            )
    
    def test_empty_dataframe_handling(self, token_representation):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame({'smiles': [], 'label': []})
        
        dataset = DatasetBase(
            dataframe=empty_df,
            representation=token_representation,
            precompute_all=True,
        )
        
        initial_size = get_vocab_size(dataset)
        
        # Should handle empty data gracefully
        extend_vocab_from_data(dataset, 'smiles')
        
        assert get_vocab_size(dataset) == initial_size
