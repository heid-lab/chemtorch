import pandas as pd
import pytest
import logging

from chemtorch.core.dataset_base import DatasetBase


def test_dataset_len_no_subsample(train_df, mock_representation):
    dataset = DatasetBase(train_df, representation=mock_representation)
    assert len(dataset) == 5
    assert len(dataset.dataframe) == 5

def test_dataset_len_with_subsample(train_df, mock_representation):
    dataset = DatasetBase(train_df, representation=mock_representation, subsample=3)
    assert len(dataset) == 3
    assert len(dataset.dataframe) == 3

def test_dataset_subsample_with_int(train_df, mock_representation):
    dataset = DatasetBase(train_df, representation=mock_representation, subsample=2)
    assert len(dataset) == 2
    assert len(dataset.dataframe) == 2

def test_dataset_subsample_with_zero(train_df, mock_representation):
    with pytest.raises(ValueError):
        DatasetBase(train_df, representation=mock_representation, subsample=0)   

def test_dataset_subsample_with_negative(train_df, mock_representation):
    with pytest.raises(ValueError):
        DatasetBase(train_df, representation=mock_representation, subsample=-1)

def test_dataset_subsample_with_float(train_df, mock_representation):
    dataset = DatasetBase(train_df, representation=mock_representation, subsample=0.4)
    assert len(dataset) == 2  # 40% of 5 is 2
    assert len(dataset.dataframe) == 2

def test_dataset_subsample_with_float_rounding(train_df, mock_representation):
    dataset = DatasetBase(train_df, representation=mock_representation, subsample=0.5)
    # 50% of 5 is 2.5 which rounds to 2 because python rounds .5 to the nearest even number (bankers rounding)
    assert len(dataset) == 2
    assert len(dataset.dataframe) == 2

def test_dataset_subsample_with_rounding_to_zero(train_df, mock_representation):
    dataset = DatasetBase(train_df, representation=mock_representation, subsample=0.01)
    assert len(dataset) == 1  # Minimum of 1 sample
    assert len(dataset.dataframe) == 1

def test_dataset_subsample_warning_for_small_fraction(train_df, mock_representation, caplog):
    """Test that a warning is logged when subsample fraction is too small."""
    with caplog.at_level(logging.WARNING):
        # Use a very small fraction that would round to 0 (0.05 * 5 = 0.25 -> rounds to 0)
        dataset = DatasetBase(train_df, representation=mock_representation, subsample=0.05)
        
    # Check that the warning was logged
    assert "Subsample fraction 0.05 too small for dataset size 5, rounding up to 1 sample." in caplog.text
    # Check that we still get 1 sample
    assert len(dataset) == 1
    assert len(dataset.dataframe) == 1

def test_dataset_subsample_no_effect(train_df, mock_representation):
    dataset = DatasetBase(train_df, representation=mock_representation, subsample=1.0)
    assert len(dataset) == 5
    assert len(dataset.dataframe) == 5

def test_dataset_basic_functionality(train_df, mock_representation):
    """Test basic dataset functionality without subsampling."""
    dataset = DatasetBase(train_df, representation=mock_representation)
    
    # Test length
    assert len(dataset) == 5
    
    # Test indexing
    item = dataset[0]
    assert item == (None, 1)  # (representation_output, label)
    
    # Test that dataframe is preserved
    assert len(dataset.dataframe) == 5
    assert list(dataset.dataframe.columns) == ["smiles", "label"]

def test_dataset_without_labels(mock_representation):
    """Test dataset functionality with no labels."""
    df_no_labels = pd.DataFrame({"smiles": ["A", "B", "C"]})
    dataset = DatasetBase(df_no_labels, representation=mock_representation)
    
    # Should return only the representation output, not a tuple
    item = dataset[0]
    assert item is None  # Just the representation output

def test_dataset_with_abstract_transform(train_df, mock_representation, mock_transform):
    """Test dataset with AbstractTransform."""
    dataset = DatasetBase(train_df, representation=mock_representation, transform=mock_transform, precompute_all=False)
    
    # Test that transform is applied
    item = dataset[0]
    assert item == ("transformed_1", 1)  # (transformed_data, label)
    
    # Test that transform is called for each access
    item2 = dataset[1]
    assert item2 == ("transformed_2", 2)  # call_count incremented
    
    # Verify transform was called
    assert mock_transform.call_count == 2

def test_dataset_with_callable_transform(train_df, mock_representation, callable_transform):
    """Test dataset with callable transform (not AbstractTransform)."""
    dataset = DatasetBase(train_df, representation=mock_representation, transform=callable_transform, precompute_all=False)
    
    # Test that transform is applied
    item = dataset[0]
    assert item == ("callable_1", 1)  # (transformed_data, label)
    
    # Test that transform is called for each access
    item2 = dataset[1]
    assert item2 == ("callable_2", 2)  # call_count incremented
    
    # Verify transform was called
    assert callable_transform.call_count == 2

def test_dataset_transform_without_labels(mock_representation, mock_transform):
    """Test transform application on dataset without labels."""
    df_no_labels = pd.DataFrame({"smiles": ["A", "B", "C"]})
    dataset = DatasetBase(df_no_labels, representation=mock_representation, transform=mock_transform, precompute_all=False)
    
    # Should return only the transformed representation output
    item = dataset[0]
    assert item == "transformed_1"  # Just the transformed output, no tuple
    
    # Verify transform was called
    assert mock_transform.call_count == 1

def test_dataset_invalid_transform(train_df, mock_representation):
    """Test that invalid transform types raise ValueError."""
    with pytest.raises(ValueError, match="Transform must be an instance of AbstractTransform, Callable, or None"):
        DatasetBase(train_df, representation=mock_representation, transform="invalid_transform")  # type: ignore

def test_dataset_none_transform(train_df, mock_representation):
    """Test dataset with None transform (no transformation)."""
    dataset = DatasetBase(train_df, representation=mock_representation, transform=None)
    
    # Test that no transform is applied
    item = dataset[0]
    assert item == (None, 1)  # (original_data, label)

def test_dataset_transform_with_precompute_disabled(train_df, mock_representation, mock_transform):
    """Test transform with precompute_all=False."""
    dataset = DatasetBase(
        train_df, 
        representation=mock_representation, 
        transform=mock_transform,
        precompute_all=False
    )
    
    # Test that transform is applied on demand
    item = dataset[0]
    assert item == ("transformed_1", 1)
    
    # Accessing same item should call transform again (no caching by default)
    item_again = dataset[0]
    assert item_again == ("transformed_2", 1)  # call_count incremented
    
    assert mock_transform.call_count == 2

def test_dataset_transform_with_precompute_enabled(train_df, mock_representation, mock_transform):
    """Test transform with precompute_all=True (default behavior)."""
    dataset = DatasetBase(
        train_df, 
        representation=mock_representation, 
        transform=mock_transform,
        precompute_all=True  # This is the default
    )
    
    # All transforms should have been called during initialization
    assert mock_transform.call_count == 5  # Called for all 5 items during precomputation
    
    # Accessing items should return precomputed results (no additional transform calls)
    initial_count = mock_transform.call_count
    item = dataset[0]
    assert item == ("transformed_1", 1)  # First precomputed item
    
    item2 = dataset[1] 
    assert item2 == ("transformed_2", 2)  # Second precomputed item
    
    # Call count should not have increased (using precomputed results)
    assert mock_transform.call_count == initial_count


