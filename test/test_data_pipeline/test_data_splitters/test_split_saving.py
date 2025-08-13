import os
import pickle

import numpy as np
import pandas as pd
import pytest

from chemtorch.components.data_pipeline.data_splitter import (
    IndexSplitter,
    RatioSplitter,
    SizeSplitter,
)


@pytest.fixture
def sample_dataframe():
    """Fixture to create a generic sample DataFrame for basic splitter tests."""
    return pd.DataFrame({"col1": range(1, 11), "col2": range(11, 21)})


@pytest.fixture
def size_splitter_dataframe():
    """
    Fixture to create a sample DataFrame for testing SizeSplitter.
    It includes a 'smiles' column with varying molecule sizes and a unique 'id'
    column to make it easy to track rows during shuffling and sorting.
    """
    # Create SMILES strings with progressively larger molecules
    smiles_data = [f"{'C' * i}>>{'C' * (i + 1)}" for i in range(1, 11)]
    return pd.DataFrame({"smiles": smiles_data, "id": range(10)})


@pytest.fixture
def index_pickle_file(tmp_path):
    """Fixture to create a temporary pickle file with pre-defined train/val/test indices."""
    split_index_path = tmp_path / "input_indices.pkl"
    indices = [np.array([0, 1, 2, 3, 4]), np.array([5, 6]), np.array([7, 8, 9])]
    with open(split_index_path, "wb") as f:
        pickle.dump([indices], f)
    return str(split_index_path)


def test_ratio_splitter_saves_indices(sample_dataframe, tmp_path):
    """Test that RatioSplitter correctly saves split indices to a file."""
    save_dir = tmp_path / "splits"
    splitter = RatioSplitter(
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        save_split_dir=str(save_dir),
        save_indices=True,
        save_csv=False,
    )

    splitter(sample_dataframe)

    indices_file = save_dir / "indices.pkl"
    assert indices_file.is_file()
    assert not (save_dir / "train.csv").exists()

    with open(indices_file, "rb") as f:
        loaded_data = pickle.load(f)
    assert isinstance(loaded_data, list) and len(loaded_data) == 1
    split_indices = loaded_data[0]
    assert isinstance(split_indices, list) and len(split_indices) == 3
    total_indices = sum(len(arr) for arr in split_indices)
    assert total_indices == len(sample_dataframe)


def test_ratio_splitter_saves_csvs(sample_dataframe, tmp_path):
    """Test that RatioSplitter correctly saves split dataframes to CSV files."""
    save_dir = tmp_path / "splits"
    splitter = RatioSplitter(
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        save_split_dir=str(save_dir),
        save_indices=False,
        save_csv=True,
    )

    data_split = splitter(sample_dataframe)

    assert (save_dir / "train.csv").is_file()
    assert (save_dir / "val.csv").is_file()
    assert (save_dir / "test.csv").is_file()
    assert not (save_dir / "indices.pkl").exists()

    loaded_train_df = pd.read_csv(save_dir / "train.csv")
    # Compare values, ignoring index differences between saved CSV and in-memory DF
    pd.testing.assert_frame_equal(
        loaded_train_df, data_split.train, check_index_type=False
    )


def test_ratio_splitter_saves_both(sample_dataframe, tmp_path):
    """Test that RatioSplitter can save both indices and CSVs simultaneously."""
    save_dir = tmp_path / "splits"
    splitter = RatioSplitter(
        save_split_dir=str(save_dir), save_indices=True, save_csv=True
    )

    splitter(sample_dataframe)

    assert (save_dir / "indices.pkl").is_file()
    assert (save_dir / "train.csv").is_file()


def test_splitter_does_not_save_by_default(sample_dataframe, tmp_path):
    """Test that the splitter does not save files when save_split_dir is None."""
    splitter = RatioSplitter()
    splitter(sample_dataframe)
    assert not any(tmp_path.iterdir()), (
        "Files were saved when they should not have been"
    )


def test_index_splitter_resaves_files(sample_dataframe, index_pickle_file, tmp_path):
    """Test that IndexSplitter can re-save the splits it creates from an index file."""
    save_dir = tmp_path / "output_splits"
    splitter = IndexSplitter(
        split_index_path=index_pickle_file,
        save_split_dir=str(save_dir),
        save_indices=True,
        save_csv=True,
    )

    splitter(sample_dataframe)

    assert (save_dir / "indices.pkl").is_file()
    assert (save_dir / "train.csv").is_file()

    with open(index_pickle_file, "rb") as f:
        original_indices = pickle.load(f)[0]
    with open(save_dir / "indices.pkl", "rb") as f:
        resaved_indices = pickle.load(f)[0]

    for original, resaved in zip(original_indices, resaved_indices):
        np.testing.assert_array_equal(original, resaved)


def test_size_splitter_saved_indices_reproduce_identical_split(
    size_splitter_dataframe, tmp_path
):
    """
    Test that data reloaded from saved indices is identical to the initially split data.

    This "round-trip" test is crucial for catching inconsistencies between the returned
    DataSplit object and the persisted data. It verifies that if the returned data is
    shuffled, the saved indices correspond to that same shuffled order, not a
    pre-shuffled state.
    """

    save_dir = tmp_path / "size_splits"
    size_splitter = SizeSplitter(
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        save_split_dir=str(save_dir),
        save_indices=True,  # round trip
        save_csv=False,
    )

    initial_data_split = size_splitter(size_splitter_dataframe)

    indices_file = save_dir / "indices.pkl"
    assert indices_file.is_file(), "Indices file was not created by SizeSplitter"

    index_splitter = IndexSplitter(split_index_path=str(indices_file))
    reloaded_data_split = index_splitter(size_splitter_dataframe)

    try:
        pd.testing.assert_frame_equal(
            initial_data_split.train, reloaded_data_split.train.reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(
            initial_data_split.val, reloaded_data_split.val.reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(
            initial_data_split.test, reloaded_data_split.test.reset_index(drop=True)
        )
    except AssertionError as e:
        pytest.fail(
            "Reloaded data does not match the initial split. This indicates a mismatch "
            "between the returned DataSplit and the saved indices, likely due to "
            f"incorrect handling of shuffling.\n\nDetails: {e}"
        )
