import pandas as pd
import pytest
import logging
from unittest.mock import Mock
from typing import cast

from chemtorch.core.data_module import DataModule
from chemtorch.utils.types import DatasetKey, DataSplit
from test.test_data_module.conftest import CallableTransform, MockTransform

@pytest.fixture
def mock_data_pipeline(train_df, val_df, test_df):
    """Mock data pipeline that returns predefined datasets."""
    mock = Mock()
    
    mock_data = DataSplit(
        train=train_df,
        val=val_df,
        test=test_df
    )
    
    # Configure the mock to return the DataSplit when called directly
    mock.return_value = mock_data
    return mock

@pytest.fixture
def mock_dataloader_factory():
    """Mock dataloader factory."""
    mock = Mock()
    return mock

def test_data_module_subsample_split_specific_train_only(mock_data_pipeline, mock_dataloader_factory, mock_representation):
    """Test that subsample dictionary only affects specified splits."""
    subsample_dict = cast(dict[DatasetKey, float], {"train": 0.6})  # Only subsample train set
    
    data_module = DataModule(
        data_pipeline=mock_data_pipeline,
        representation=mock_representation,
        dataloader_factory=mock_dataloader_factory,
        subsample=subsample_dict
    )
    
    # Train dataset should be subsampled (60% of 5 = 3)
    assert len(data_module.train_dataset) == 3
    
    # Val and test datasets should not be affected
    assert len(data_module.val_dataset) == 2
    assert len(data_module.test_dataset) == 2  # test_dataset is a single dataset for simple case

def test_data_module_subsample_split_specific_multiple_splits(mock_data_pipeline, mock_dataloader_factory, mock_representation):
    """Test subsample dictionary affecting multiple splits."""
    subsample_dict = cast(dict[DatasetKey, float], {"train": 0.6, "val": 0.5})  # Subsample both train and val
    
    data_module = DataModule(
        data_pipeline=mock_data_pipeline,
        representation=mock_representation,
        dataloader_factory=mock_dataloader_factory,
        subsample=subsample_dict
    )
    
    # Train dataset should be subsampled (60% of 5 = 3)
    assert len(data_module.train_dataset) == 3
    
    # Val dataset should be subsampled (50% of 2 = 1)
    assert len(data_module.val_dataset) == 1
    
    # Test dataset should not be affected
    assert len(data_module.test_dataset) == 2

def test_data_module_subsample_unaffected_split(mock_data_pipeline, mock_dataloader_factory, mock_representation):
    """Test that splits not specified in subsample dict are unaffected."""
    subsample_dict = cast(dict[DatasetKey, float], {"train": 0.4})  # Only affect train
    
    data_module = DataModule(
        data_pipeline=mock_data_pipeline,
        representation=mock_representation,
        dataloader_factory=mock_dataloader_factory,
        subsample=subsample_dict
    )
    
    # Train dataset should be subsampled (40% of 5 = 2)
    assert len(data_module.train_dataset) == 2
    
    # Val and test should remain unchanged
    assert len(data_module.val_dataset) == 2
    assert len(data_module.test_dataset) == 2

def test_data_module_subsample_global_value(mock_data_pipeline, mock_dataloader_factory, mock_representation):
    """Test that a single subsample value affects all splits."""
    data_module = DataModule(
        data_pipeline=mock_data_pipeline,
        representation=mock_representation,
        dataloader_factory=mock_dataloader_factory,
        subsample=0.5  # Global subsample
    )
    
    # All datasets should be subsampled
    assert len(data_module.train_dataset) == 2  # 50% of 5 = 2 (rounds to 2)
    assert len(data_module.val_dataset) == 1   # 50% of 2 = 1
    assert len(data_module.test_dataset) == 1  # 50% of 2 = 1

def test_data_module_subsample_dict_validation(mock_data_pipeline, mock_dataloader_factory, mock_representation):
    """Test that invalid subsample dictionary keys raise ValueError."""
    # Use a proper type but invalid key content
    invalid_subsample = {"invalid_split": 0.5}  # type: ignore
    
    with pytest.raises(ValueError, match="Subsample dictionary keys must be one of"):
        DataModule(
            data_pipeline=mock_data_pipeline,
            representation=mock_representation,
            dataloader_factory=mock_dataloader_factory,
            subsample=invalid_subsample  # type: ignore
        )

def test_data_module_no_subsample(mock_data_pipeline, mock_dataloader_factory, mock_representation):
    """Test data module without any subsampling."""
    data_module = DataModule(
        data_pipeline=mock_data_pipeline,
        representation=mock_representation,
        dataloader_factory=mock_dataloader_factory,
        subsample=None
    )
    
    # All datasets should have their original sizes
    assert len(data_module.train_dataset) == 5
    assert len(data_module.val_dataset) == 2
    assert len(data_module.test_dataset) == 2

def test_data_module_single_transform_all_splits(mock_data_pipeline, mock_dataloader_factory, mock_representation, mock_transform):
    """Test data module with a single transform applied to all splits."""
    data_module = DataModule(
        data_pipeline=mock_data_pipeline,
        representation=mock_representation,
        dataloader_factory=mock_dataloader_factory,
        transform=mock_transform
    )
    
    # Test that all datasets have the transform applied
    train_item = data_module.train_dataset[0]
    assert train_item[0].startswith("transformed_")  # transformed data
    
    val_item = data_module.val_dataset[0]
    assert val_item[0].startswith("transformed_")
    
    test_item = data_module.test_dataset[0]
    assert test_item[0].startswith("transformed_")

def test_data_module_no_transform(mock_data_pipeline, mock_dataloader_factory, mock_representation):
    """Test data module without any transforms."""
    data_module = DataModule(
        data_pipeline=mock_data_pipeline,
        representation=mock_representation,
        dataloader_factory=mock_dataloader_factory,
        transform=None
    )
    
    # Test that no transforms are applied
    train_item = data_module.train_dataset[0]
    assert train_item == (None, 1)  # No transform applied
    
    val_item = data_module.val_dataset[0]
    assert val_item == (None, 6)
    
    test_item = data_module.test_dataset[0]
    assert test_item == (None, 8)

def test_data_module_dict_transform_single_split(mock_data_pipeline, mock_dataloader_factory, mock_representation, mock_transform):
    """Test data module with dictionary transform for single split."""
    transforms = cast(dict[DatasetKey, MockTransform], {"test": mock_transform})  # Only apply transform to test set  # type: ignore
    
    data_module = DataModule(
        data_pipeline=mock_data_pipeline,
        representation=mock_representation,
        dataloader_factory=mock_dataloader_factory,
        transform=transforms  # type: ignore
    )
    
    # Train and val should have no transforms
    train_item = data_module.train_dataset[0]
    assert train_item == (None, 1)
    
    val_item = data_module.val_dataset[0]
    assert val_item == (None, 6)
    
    # Test should have transform applied
    test_item = data_module.test_dataset[0]
    assert test_item[0].startswith("transformed_")

def test_data_module_dict_transform_multiple_splits(mock_data_pipeline, mock_dataloader_factory, mock_representation, mock_transform):
    """Test data module with dictionary transforms for multiple splits."""
    
    transforms = cast(dict[DatasetKey, MockTransform], {
        "train": MockTransform("train_transform"),
        "val": MockTransform("val_transform"),
        "test": MockTransform("test_transform")
    })  # type: ignore
    
    data_module = DataModule(
        data_pipeline=mock_data_pipeline,
        representation=mock_representation,
        dataloader_factory=mock_dataloader_factory,
        transform=transforms  # type: ignore
    )
    
    # Each split should use its specific transform
    train_item = data_module.train_dataset[0]
    assert train_item[0] == "train_transform_1"
    
    val_item = data_module.val_dataset[0]
    assert val_item[0] == "val_transform_1"
    
    test_item = data_module.test_dataset[0]
    assert test_item[0] == "test_transform_1"

def test_data_module_test_transform_list(mock_data_pipeline, mock_dataloader_factory, mock_representation):
    """Test data module with list of transforms for test set."""
    
    test_transforms = [
        MockTransform("test_transform_1"),
        MockTransform("test_transform_2"),
    ]
    
    transforms = cast(dict[DatasetKey, list[MockTransform]], {"test": test_transforms})  # type: ignore
    
    data_module = DataModule(
        data_pipeline=mock_data_pipeline,
        representation=mock_representation,
        dataloader_factory=mock_dataloader_factory,
        transform=transforms  # type: ignore
    )
    
    # Should have main test dataset plus additional test datasets for each transform
    assert hasattr(data_module, 'test_dataset')
    assert hasattr(data_module, 'test_1_dataset')
    assert hasattr(data_module, 'test_2_dataset')
    
    # Main test dataset (no transform)
    test_item_main = data_module.get_dataset("test")[0]
    assert test_item_main == (None, 8)  # No transform
    
    # First additional test dataset (first transform)
    test_item_1 = data_module.get_dataset("test_1")[0]
    assert test_item_1[0] == "test_transform_1_1"
    
    # Second additional test dataset (second transform)
    test_item_2 = data_module.get_dataset("test_2")[0]
    assert test_item_2[0] == "test_transform_2_1"
    
    # Test dataloader names mapping should return proper indices
    names = data_module.maybe_get_test_dataloader_idx_to_suffix()
    assert names == {1: "1", 2: "2"}  # index 0 is main test, 1 and 2 are additional

def test_data_module_test_transform_dict(mock_data_pipeline, mock_dataloader_factory, mock_representation):
    """Test data module with dict of named transforms for test set."""
    
    test_transforms = {
        "augmented": MockTransform("augmented"),
        "normalized": MockTransform("normalized"),
    }
    
    transforms = cast(dict[DatasetKey, dict[str, MockTransform]], {"test": test_transforms})  # type: ignore
    
    data_module = DataModule(
        data_pipeline=mock_data_pipeline,
        representation=mock_representation,
        dataloader_factory=mock_dataloader_factory,
        transform=transforms  # type: ignore
    )
    
    # Should have main test dataset plus additional named test datasets
    assert hasattr(data_module, 'test_dataset')
    assert hasattr(data_module, 'test_augmented_dataset')
    assert hasattr(data_module, 'test_normalized_dataset')
    
    # Main test dataset (no transform)
    test_item_main = data_module.get_dataset("test")[0]
    assert test_item_main == (None, 8)
    
    # Augmented test dataset
    test_item_aug = data_module.get_dataset("test_augmented")[0]
    assert test_item_aug[0] == "augmented_1"
    
    # Normalized test dataset
    test_item_norm = data_module.get_dataset("test_normalized")[0]
    assert test_item_norm[0] == "normalized_1"
    
    # Test getting test dataset names mapping
    names = data_module.maybe_get_test_dataloader_idx_to_suffix()
    assert names == {1: "augmented", 2: "normalized"}  # index 0 is main test

def test_data_module_test_dataloader_multiple_list(mock_data_pipeline, mock_dataloader_factory, mock_representation):
    """Test that multiple test datasets from list transforms create multiple dataloaders."""
    
    test_transforms = [MockTransform("transform1"), MockTransform("transform2")]
    transforms = cast(dict[DatasetKey, list[MockTransform]], {"test": test_transforms})  # type: ignore
    
    data_module = DataModule(
        data_pipeline=mock_data_pipeline,
        representation=mock_representation,
        dataloader_factory=mock_dataloader_factory,
        transform=transforms  # type: ignore
    )
    
    # Test dataloader should return a list
    test_dataloaders = data_module.test_dataloader()
    assert isinstance(test_dataloaders, list)
    assert len(test_dataloaders) == 3  # test + test_1 + test_2

def test_data_module_test_dataloader_multiple_dict(mock_data_pipeline, mock_dataloader_factory, mock_representation):
    """Test that multiple test datasets from dict transforms create multiple dataloaders."""
    
    test_transforms = {"aug1": MockTransform("aug1"), "aug2": MockTransform("aug2")}
    transforms = cast(dict[DatasetKey, dict[str, MockTransform]], {"test": test_transforms})  # type: ignore
    
    data_module = DataModule(
        data_pipeline=mock_data_pipeline,
        representation=mock_representation,
        dataloader_factory=mock_dataloader_factory,
        transform=transforms  # type: ignore
    )
    
    # Test dataloader should return a list
    test_dataloaders = data_module.test_dataloader()
    assert isinstance(test_dataloaders, list)
    assert len(test_dataloaders) == 3  # test + test_aug1 + test_aug2

def test_data_module_callable_transforms(mock_data_pipeline, mock_dataloader_factory, mock_representation):
    """Test data module with callable transforms."""
    
    transforms = cast(dict[DatasetKey, CallableTransform], {
        "train": CallableTransform("train_callable"),
        "val": CallableTransform("val_callable"),
    })  # type: ignore
    
    data_module = DataModule(
        data_pipeline=mock_data_pipeline,
        representation=mock_representation,
        dataloader_factory=mock_dataloader_factory,
        transform=transforms  # type: ignore
    )
    
    # Test that callable transforms work
    train_item = data_module.train_dataset[0]
    assert train_item[0] == "train_callable_1"
    
    val_item = data_module.val_dataset[0]
    assert val_item[0] == "val_callable_1"
    
    # Test dataset should have no transform (not specified)
    test_item = data_module.test_dataset[0]
    assert test_item == (None, 8)  # No transform applied

def test_data_module_mixed_callable_and_transform_types(mock_data_pipeline, mock_dataloader_factory, mock_representation, mock_transform):
    """Test data module with mixed callable and AbstractTransform types."""
    
    transforms = cast(dict[DatasetKey, CallableTransform | MockTransform], {
        "train": CallableTransform("train_callable"),  # Callable
        "val": mock_transform,  # AbstractTransform
    })  # type: ignore
    
    data_module = DataModule(
        data_pipeline=mock_data_pipeline,
        representation=mock_representation,
        dataloader_factory=mock_dataloader_factory,
        transform=transforms  # type: ignore
    )
    
    # Test that both types work
    train_item = data_module.train_dataset[0]
    assert train_item[0] == "train_callable_1"
    
    val_item = data_module.val_dataset[0]
    assert val_item[0].startswith("transformed_")

# Edge case tests

def test_data_module_invalid_transform_dict_keys(mock_data_pipeline, mock_dataloader_factory, mock_representation):
    """Test that invalid transform dictionary keys raise ValueError."""
    
    invalid_transforms = {
        "invalid_split": MockTransform("invalid"),
        "train": MockTransform("train"),
    }
    
    with pytest.raises(ValueError, match="Transforms dictionary keys must be one of"):
        DataModule(
            data_pipeline=mock_data_pipeline,
            representation=mock_representation,
            dataloader_factory=mock_dataloader_factory,
            transform=invalid_transforms  # type: ignore
        )

def test_data_module_invalid_transform_type(mock_data_pipeline, mock_dataloader_factory, mock_representation):
    """Test that invalid transform types raise TypeError."""
    with pytest.raises(TypeError, match="Transforms must be either an AbstractTransform instance, callable, or a dictionary"):
        DataModule(
            data_pipeline=mock_data_pipeline,
            representation=mock_representation,
            dataloader_factory=mock_dataloader_factory,
            transform="invalid_transform"  # type: ignore
        )

def test_data_module_invalid_train_val_predict_transform_type(mock_data_pipeline, mock_dataloader_factory, mock_representation):
    """Test that invalid transform types for train/val/predict keys raise TypeError."""
    invalid_transforms = {
        "train": ["not_allowed_list"],  # Lists only allowed for test key
    }
    
    with pytest.raises(TypeError, match="The value for the 'train' key in transforms must be an AbstractTransform or callable"):
        DataModule(
            data_pipeline=mock_data_pipeline,
            representation=mock_representation,
            dataloader_factory=mock_dataloader_factory,
            transform=invalid_transforms  # type: ignore
        )

def test_data_module_invalid_test_transform_list_contents(mock_data_pipeline, mock_dataloader_factory, mock_representation):
    """Test that invalid contents in test transform list raise TypeError."""
    invalid_test_transforms = [
        "not_a_transform",  # Invalid list element
    ]
    
    transforms = {"test": invalid_test_transforms}
    
    with pytest.raises(TypeError, match="All elements in the list or dict of transforms for the 'test' key must be AbstractTransform instances or callables"):
        DataModule(
            data_pipeline=mock_data_pipeline,
            representation=mock_representation,
            dataloader_factory=mock_dataloader_factory,
            transform=transforms  # type: ignore
        )

def test_data_module_invalid_test_transform_dict_contents(mock_data_pipeline, mock_dataloader_factory, mock_representation):
    """Test that invalid contents in test transform dict raise TypeError."""
    invalid_test_transforms = {
        "valid": lambda x: x,
        "invalid": "not_a_transform",  # Invalid dict value
    }
    
    transforms = {"test": invalid_test_transforms}
    
    with pytest.raises(TypeError, match="All elements in the list or dict of transforms for the 'test' key must be AbstractTransform instances or callables"):
        DataModule(
            data_pipeline=mock_data_pipeline,
            representation=mock_representation,
            dataloader_factory=mock_dataloader_factory,
            transform=transforms  # type: ignore
        )

def test_data_module_invalid_test_transform_type(mock_data_pipeline, mock_dataloader_factory, mock_representation):
    """Test that invalid test transform type raises TypeError."""
    transforms = {"test": 123}  # Invalid type for test transform
    
    with pytest.raises(TypeError, match="The value for the 'test' key in transforms must be an AbstractTransform, callable, a list of AbstractTransforms, or a dict of AbstractTransforms"):
        DataModule(
            data_pipeline=mock_data_pipeline,
            representation=mock_representation,
            dataloader_factory=mock_dataloader_factory,
            transform=transforms  # type: ignore
        )

def test_data_module_empty_test_transform_list(mock_data_pipeline, mock_dataloader_factory, mock_representation):
    """Test data module with empty test transform list."""
    transforms = cast(dict[DatasetKey, list], {"test": []})  # Empty list  # type: ignore
    
    data_module = DataModule(
        data_pipeline=mock_data_pipeline,
        representation=mock_representation,
        dataloader_factory=mock_dataloader_factory,
        transform=transforms  # type: ignore
    )
    
    # Should create only the original test dataset
    assert hasattr(data_module, 'test_dataset')
    # Should not have any additional test datasets
    assert not hasattr(data_module, 'test_1_dataset')
    
    test_item = data_module.get_dataset("test")[0]
    assert test_item == (None, 8)  # No transform
    
    # Names should return None for single test dataset
    names = data_module.maybe_get_test_dataloader_idx_to_suffix()
    assert names is None

def test_data_module_empty_test_transform_dict(mock_data_pipeline, mock_dataloader_factory, mock_representation):
    """Test data module with empty test transform dict."""
    transforms = cast(dict[DatasetKey, dict], {"test": {}})  # Empty dict  # type: ignore
    
    data_module = DataModule(
        data_pipeline=mock_data_pipeline,
        representation=mock_representation,
        dataloader_factory=mock_dataloader_factory,
        transform=transforms  # type: ignore
    )
    
    # Should create only the original test dataset
    assert hasattr(data_module, 'test_dataset')
    # Should not have any additional test datasets
    dataset_names = data_module.get_dataset_names()
    test_dataset_names = [name for name in dataset_names if name.startswith("test")]
    assert test_dataset_names == ["test"]  # Only original dataset
    
    test_item = data_module.get_dataset("test")[0]
    assert test_item == (None, 8)  # No transform
    
    # Names should return None for single test dataset
    names = data_module.maybe_get_test_dataloader_idx_to_suffix()
    assert names is None
