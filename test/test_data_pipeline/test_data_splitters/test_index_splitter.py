import pickle

import pandas as pd
import pytest

from chemtorch.components.data_pipeline.data_splitter import IndexSplitter
from chemtorch.utils import DataSplit


@pytest.fixture
def index_pickle_file(tmp_path):
    """Fixture to create a temporary pickle file with train, val, and test indices."""
    split_index_path = tmp_path / "indices.pkl"
    # The IndexSplitter expects a pickle file containing a tuple/list, where the first element is a list of 3 arrays
    indices = [[0, 1, 2, 3, 4], [5, 6], [7, 8, 9]]  # Train, val, test indices
    with open(split_index_path, "wb") as f:
        pickle.dump([indices], f)
    return str(split_index_path)


@pytest.fixture
def invalid_index_pickle_file(tmp_path):
    """Fixture to create an invalid pickle file with incorrect indices."""
    split_index_path = tmp_path / "invalid_indices.pkl"
    indices = [[0, 1, 2], [3, 4]]  # Only 2 splits instead of 3
    with open(split_index_path, "wb") as f:
        pickle.dump([indices], f)
    return str(split_index_path)


@pytest.fixture
def malformed_pickle_file(tmp_path):
    """Fixture to create a malformed pickle file (not a list/tuple)."""
    split_index_path = tmp_path / "malformed.pkl"
    with open(split_index_path, "wb") as f:
        pickle.dump("not a list", f)
    return str(split_index_path)


def test_index_splitter(sample_dataframe, index_pickle_file):
    """Test the IndexSplitter functionality."""
    splitter = IndexSplitter(split_index_path=index_pickle_file)
    data_split = splitter.__call__(sample_dataframe)

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
        IndexSplitter(split_index_path=invalid_index_pickle_file)


def test_index_splitter_empty_dataframe(index_pickle_file):
    """Test IndexSplitter with an empty DataFrame."""
    splitter = IndexSplitter(split_index_path=index_pickle_file)
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        splitter.__call__(empty_df)


def test_index_splitter_malformed_pickle(malformed_pickle_file):
    """Test IndexSplitter with a malformed pickle file."""
    with pytest.raises(ValueError, match="Pickle file must contain exactly 3 arrays"):
        IndexSplitter(split_index_path=malformed_pickle_file)


def test_index_splitter_out_of_bounds_indices(index_pickle_file, sample_dataframe):
    """Test IndexSplitter with indices that are out of bounds."""
    # Overwrite the pickle file with out-of-bounds indices
    with open(index_pickle_file, "wb") as f:
        pickle.dump([[[0, 1, 2, 100], [5, 6], [7, 8, 9]]], f)
    splitter = IndexSplitter(split_index_path=index_pickle_file)
    with pytest.raises(IndexError):
        splitter.__call__(sample_dataframe)


def test_index_splitter_duplicate_indices(tmp_path, sample_dataframe):
    """Test IndexSplitter with duplicate indices across splits."""
    split_index_path = tmp_path / "dup_indices.pkl"
    indices = [
        [0, 1, 2],
        [2, 3],
        [4, 5],
    ]  # Index 2 appears in both train and val
    with open(split_index_path, "wb") as f:
        pickle.dump([indices], f)
    splitter = IndexSplitter(split_index_path=str(split_index_path))
    data_split = splitter.__call__(sample_dataframe)
    # Check that index 2 appears in both train and val splits
    assert 2 in data_split.train.index.values
    assert 2 in data_split.val.index.values
    assert isinstance(data_split.test, pd.DataFrame)

    # Check that the splits are not empty
    assert not data_split.train.empty
    assert not data_split.val.empty
    assert not data_split.test.empty

    # Check that the splits contain the correct rows
    assert data_split.train.equals(sample_dataframe.iloc[[0, 1, 2]])
    assert data_split.val.equals(sample_dataframe.iloc[[2, 3]])
    assert data_split.test.equals(sample_dataframe.iloc[[4, 5]])

    # Check that the total number of rows matches the input (allowing for duplicates)
    total_rows = len(data_split.train) + len(data_split.val) + len(data_split.test)
    assert total_rows == 7


def test_index_splitter_invalid_pickle(invalid_index_pickle_file):
    """Test IndexSplitter with an invalid pickle file."""
    with pytest.raises(ValueError, match="Pickle file must contain exactly 3 arrays"):
        IndexSplitter(split_index_path=invalid_index_pickle_file)


def test_index_splitter_empty_dataframe(index_pickle_file):
    """Test IndexSplitter with an empty DataFrame."""
    splitter = IndexSplitter(split_index_path=index_pickle_file)
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        splitter.__call__(empty_df)
