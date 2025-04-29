import pickle
import pandas as pd
import pytest
from deeprxn.data import DataSplit
from deeprxn.data_splitter.index_splitter import IndexSplitter


@pytest.fixture
def index_pickle_file(tmp_path):
    """Fixture to create a temporary pickle file with train, val, and test indices."""
    index_path = tmp_path / "indices.pkl"
    indices = [[0, 1, 2, 3, 4], [5, 6], [7, 8, 9]]  # Train, val, test indices
    with open(index_path, "wb") as f:
        pickle.dump(indices, f)
    return str(index_path)


@pytest.fixture
def invalid_index_pickle_file(tmp_path):
    """Fixture to create an invalid pickle file with incorrect indices."""
    index_path = tmp_path / "invalid_indices.pkl"
    indices = [[0, 1, 2], [3, 4]]  # Only 2 splits instead of 3
    with open(index_path, "wb") as f:
        pickle.dump(indices, f)
    return str(index_path)


def test_index_splitter(sample_dataframe, index_pickle_file):
    """Test the IndexSplitter functionality."""
    splitter = IndexSplitter(index_path=index_pickle_file)
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

    # Check that the splits contain the correct rows
    assert data_split.train.equals(sample_dataframe.iloc[[0, 1, 2, 3, 4]])
    assert data_split.val.equals(sample_dataframe.iloc[[5, 6]])
    assert data_split.test.equals(sample_dataframe.iloc[[7, 8, 9]])

    # Check that the total number of rows matches the input
    total_rows = len(data_split.train) + len(data_split.val) + len(data_split.test)
    assert total_rows == len(sample_dataframe)


def test_index_splitter_invalid_pickle(invalid_index_pickle_file):
    """Test IndexSplitter with an invalid pickle file."""
    with pytest.raises(ValueError, match="Pickle file must contain exactly 3 arrays"):
        IndexSplitter(index_path=invalid_index_pickle_file)


def test_index_splitter_empty_dataframe(index_pickle_file):
    """Test IndexSplitter with an empty DataFrame."""
    splitter = IndexSplitter(index_path=index_pickle_file)
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        splitter.forward(empty_df)