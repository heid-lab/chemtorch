import pytest
import pandas as pd
from deeprxn.data_pipeline.data_splitter.ratio_splitter import RatioSplitter
from deeprxn.data_pipeline.data_pipeline import DataSplit

def test_ratio_splitter(sample_dataframe):
    """Test the RatioSplitter functionality."""
    splitter = RatioSplitter(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
    data_split = splitter.forward(sample_dataframe)

    # Check that the output is a DataSplit object
    assert isinstance(data_split, DataSplit)

    # Check that the splits are DataFrames
    assert isinstance(data_split.train, pd.DataFrame)
    assert isinstance(data_split.val, pd.DataFrame)
    assert isinstance(data_split.test, pd.DataFrame)

    # Check that the splits are not empty
    assert not data_split.train.empty
    assert not data_split.val.empty
    assert not data_split.test.empty

    # Check that the total number of rows matches the input
    total_rows = len(data_split.train) + len(data_split.val) + len(data_split.test)
    assert total_rows == len(sample_dataframe)

    # Check that the ratios are approximately correct
    assert len(data_split.train) == pytest.approx(0.7 * len(sample_dataframe), abs=1)
    assert len(data_split.val) == pytest.approx(0.2 * len(sample_dataframe), abs=1)
    assert len(data_split.test) == pytest.approx(0.1 * len(sample_dataframe), abs=1)

def test_ratio_splitter_invalid_ratios():
    """Test RatioSplitter with invalid ratios."""
    with pytest.raises(ValueError, match="Ratios must sum to 1"):
        RatioSplitter(train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)  # Sum > 1

    with pytest.raises(ValueError, match="Ratios must sum to 1"):
        RatioSplitter(train_ratio=0.5, val_ratio=0.5, test_ratio=0.2)  # Sum > 1

    with pytest.raises(ValueError, match="Ratios must sum to 1"):
        RatioSplitter(train_ratio=0.5, val_ratio=0.4, test_ratio=0.0)  # Sum < 1

def test_ratio_splitter_empty_dataframe():
    """Test RatioSplitter with an empty DataFrame."""
    splitter = RatioSplitter(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        splitter.forward(empty_df)