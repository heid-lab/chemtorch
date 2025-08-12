"""
Test the DataModule dataset operations to ensure they only operate on initialized datasets.
"""

import pytest
import pandas as pd
from unittest.mock import Mock

from chemtorch.core.data_module import DataModule
from chemtorch.utils.types import DataSplit


def test_data_module_only_applies_to_initialized_datasets():
    """Test that dataset operations only apply to initialized datasets."""
    
    # Create sample data
    sample_df = pd.DataFrame({
        'smiles': ['CCO', 'CCC', 'CCCC'],
        'label': [1, 2, 3]
    })
    
    # Mock data pipeline that returns DataSplit (train/val/test only, no predict)
    def mock_data_pipeline():
        return DataSplit(
            train=sample_df.iloc[:1],
            val=sample_df.iloc[1:2],
            test=sample_df.iloc[2:3],
        )
    
    # Mock dataset factory
    def mock_dataset_factory(df):
        mock_dataset = Mock()
        mock_dataset.dataframe = df
        return mock_dataset
    
    # Mock dataloader factory
    def mock_dataloader_factory(dataset, shuffle):
        return Mock()
    
    # Create DataModule
    data_module = DataModule(
        data_pipeline=mock_data_pipeline,
        dataset_factory=mock_dataset_factory,
        dataloader_factory=mock_dataloader_factory,
    )
    
    # Verify only train, val, test datasets are initialized
    assert set(data_module.datasets.keys()) == {'train', 'val', 'test'}
    assert 'predict' not in data_module.datasets
    
    # Create an operation that counts how many times it's called
    call_count = 0
    def count_operation(dataset):
        nonlocal call_count
        call_count += 1
    
    # Apply the operation
    data_module.apply_dataset_operation(count_operation)
    
    # Should only be called 3 times (once for each initialized dataset)
    assert call_count == 3


def test_data_module_with_predict_only():
    """Test DataModule when only predict dataset is initialized."""
    
    sample_df = pd.DataFrame({
        'smiles': ['CCO', 'CCC'],
        'label': [1, 2]
    })
    
    # Mock data pipeline that returns DataFrame (predict only)
    def mock_data_pipeline():
        return sample_df
    
    def mock_dataset_factory(df):
        mock_dataset = Mock()
        mock_dataset.dataframe = df
        return mock_dataset
    
    def mock_dataloader_factory(dataset, shuffle):
        return Mock()
    
    data_module = DataModule(
        data_pipeline=mock_data_pipeline,
        dataset_factory=mock_dataset_factory,
        dataloader_factory=mock_dataloader_factory,
    )
    
    # Verify only predict dataset is initialized
    assert set(data_module.datasets.keys()) == {'predict'}
    assert 'train' not in data_module.datasets
    assert 'val' not in data_module.datasets
    assert 'test' not in data_module.datasets
    
    # Create an operation
    call_count = 0
    def count_operation(dataset):
        nonlocal call_count
        call_count += 1
    
    # Apply the operation
    data_module.apply_dataset_operation(count_operation)
    
    # Should only be called once (for predict dataset)
    assert call_count == 1
