import pytest
import pandas as pd
from deeprxn.data_splitter.ratio_splitter import RatioSplitter
from deeprxn.data import DataSplit

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