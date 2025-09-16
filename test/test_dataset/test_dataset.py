import pandas as pd
import pytest
import logging

from chemtorch.components.dataset.dataset_base import DatasetBase
from chemtorch.components.representation.abstract_representation import AbstractRepresentation


class NoOpMockRepresentation(AbstractRepresentation[None]):
    def construct(self, smiles: str) -> None:
        return None

@pytest.fixture
def train_df():
    return pd.DataFrame({"smiles": ["A", "B", "C", "D", "E"], "label": [1, 2, 3, 4, 5]})

@pytest.fixture
def val_df():
    return pd.DataFrame({"smiles": ["F", "G"], "label": [6, 7]})

@pytest.fixture
def test_df():
    return pd.DataFrame({"smiles": ["H", "I"], "label": [8, 9]})

def test_dataset_len_no_subsample(train_df):
    train_dataset = DatasetBase(train_df, split="train", representation=NoOpMockRepresentation())
    assert len(train_dataset) == 5
    assert len(train_dataset.dataframe) == 5

def test_dataset_len_with_subsample(train_df):
    train_dataset = DatasetBase(train_df, split="train", representation=NoOpMockRepresentation(), subsample=3)
    assert len(train_dataset) == 3
    assert len(train_dataset.dataframe) == 3

def test_dataset_subsample_with_int(train_df):
    train_dataset = DatasetBase(train_df, split="train", representation=NoOpMockRepresentation(), subsample=2)
    assert len(train_dataset) == 2
    assert len(train_dataset.dataframe) == 2

def test_dataset_subsample_with_zero(train_df):
    with pytest.raises(ValueError):
        DatasetBase(train_df, split="train", representation=NoOpMockRepresentation(), subsample=0)   

def test_dataset_subsample_with_negative(train_df):
    with pytest.raises(ValueError):
        DatasetBase(train_df, split="train", representation=NoOpMockRepresentation(), subsample=-1)

def test_dataset_subsample_with_float(train_df):
    train_dataset = DatasetBase(train_df, split="train", representation=NoOpMockRepresentation(), subsample=0.4)
    assert len(train_dataset) == 2  # 40% of 5 is 2
    assert len(train_dataset.dataframe) == 2

def test_dataset_subsample_with_float_rounding(train_df):
    train_dataset = DatasetBase(train_df, split="train", representation=NoOpMockRepresentation(), subsample=0.5)
    # 50% of 5 is 2.5 which rounds to 2 because python rounds .5 to the nearest even number (bankers rounding)
    assert len(train_dataset) == 2
    assert len(train_dataset.dataframe) == 2

def test_dataset_subsample_with_rounding_to_zero(train_df):
    train_dataset = DatasetBase(train_df, split="train", representation=NoOpMockRepresentation(), subsample=0.01)
    assert len(train_dataset) == 1  # Minimum of 1 sample
    assert len(train_dataset.dataframe) == 1

def test_dataset_subsample_warning_for_small_fraction(train_df, caplog):
    """Test that a warning is logged when subsample fraction is too small."""
    with caplog.at_level(logging.WARNING):
        # Use a very small fraction that would round to 0 (0.05 * 5 = 0.25 -> rounds to 0)
        train_dataset = DatasetBase(train_df, split="train", representation=NoOpMockRepresentation(), subsample=0.05)
        
    # Check that the warning was logged
    assert "Subsample fraction 0.05 too small for dataset size 5, rounding up to 1 sample." in caplog.text
    # Check that we still get 1 sample
    assert len(train_dataset) == 1
    assert len(train_dataset.dataframe) == 1

def test_dataset_subsample_no_effect(train_df):
    train_dataset = DatasetBase(train_df, split="train", representation=NoOpMockRepresentation(), subsample=1.0)
    assert len(train_dataset) == 5
    assert len(train_dataset.dataframe) == 5

def test_dataset_subsample_split(train_df):
    train_dataset = DatasetBase(train_df, split="train", representation=NoOpMockRepresentation(), subsample=0.6, subsample_splits=["train"])
    assert len(train_dataset) == 3  # 60% of 5 is 3
    assert len(train_dataset.dataframe) == 3

def test_dataset_subsample_unaffected_split(val_df):
    val_df = DatasetBase(val_df, split="val", representation=NoOpMockRepresentation(), subsample=0.6, subsample_splits=["train"])
    assert len(val_df) == 2
    assert len(val_df.dataframe) == 2

def test_dataset_subsample_multiple_splits(train_df, val_df):
    train_dataset = DatasetBase(train_df, split="train", representation=NoOpMockRepresentation(), subsample=0.6, subsample_splits=["train", "val"])
    assert len(train_dataset) == 3  # 60% of 5 is 3
    assert len(train_dataset.dataframe) == 3

    val_dataset = DatasetBase(val_df, split="val", representation=NoOpMockRepresentation(), subsample=0.6, subsample_splits=["train", "val"])
    assert len(val_dataset) == 1  # 60% of 2 = 1.2, round() rounds to 1
    assert len(val_dataset.dataframe) == 1


