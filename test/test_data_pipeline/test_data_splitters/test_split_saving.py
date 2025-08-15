import pandas as pd
import pytest
import pickle
import os
from pathlib import Path

from chemtorch.components.data_pipeline.data_splitter import (
    IndexSplitter,
    RatioSplitter,
    ScaffoldSplitter,
    SizeSplitter,
    TargetSplitter,
)
from chemtorch.utils import DataSplit


def assert_frames_are_identical(
    df1: pd.DataFrame, df2: pd.DataFrame, splitter_name: str
):
    """
    Asserts that two DataFrames are identical in content and row order.
    The index is reset on both before comparison.
    """
    try:
        # Resetting index on both sides makes this a pure content + order comparison
        pd.testing.assert_frame_equal(
            df1.reset_index(drop=True),
            df2.reset_index(drop=True),
        )
    except AssertionError as e:
        pytest.fail(
            f"DataFrames from {splitter_name} are not identical after a save/load round-trip. "
            "This indicates a mismatch between the returned DataFrame and the saved indices.\n\n"
            f"Details: {e}"
        )


@pytest.mark.parametrize(
    "splitter_class, splitter_args, dataframe_fixture, required_fixture_arg",
    [
        (IndexSplitter, {}, "sample_dataframe", "index_pickle_file"),
        (
            RatioSplitter,
            {"train_ratio": 0.7, "val_ratio": 0.2, "test_ratio": 0.1},
            "sample_dataframe",
            None,
        ),
        (
            ScaffoldSplitter,
            {"train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1},
            "scaffold_dataframe",
            None,
        ),
        (
            SizeSplitter,
            {"train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1},
            "size_splitter_dataframe",
            None,
        ),
        (
            TargetSplitter,
            {"train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1},
            "target_splitter_dataframe",
            None,
        ),
    ],
)
def test_splitter_saving_and_reloading_consistency(
    splitter_class,
    splitter_args,
    dataframe_fixture,
    required_fixture_arg,
    request,
    tmp_path,
):
    """
    Tests that splitters correctly save files that can be used to perfectly
    reconstruct the original data splits.
    """
    # --- 1. Setup ---
    original_df = request.getfixturevalue(dataframe_fixture)
    current_args = splitter_args.copy()

    if required_fixture_arg:
        current_args["split_index_path"] = request.getfixturevalue(required_fixture_arg)

    save_dir = tmp_path / "split_output"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Use new API: save_path instead of save_split_dir
    current_args["save_path"] = str(save_dir / "indices.pkl")
    
    splitter = splitter_class(**current_args)

    # --- 2. Action ---
    initial_data_split = splitter(original_df)

    # --- 3. Verification: Round-trip via saved indices ---
    indices_path = save_dir / "indices.pkl"
    assert indices_path.exists(), f"indices.pkl not found for {splitter_class.__name__}"

    reloader_splitter = IndexSplitter(split_index_path=str(indices_path))
    reloaded_data_split = reloader_splitter(original_df)

    # This helper function handles index differences correctly.
    splitter_name = splitter_class.__name__
    assert_frames_are_identical(
        initial_data_split.train, reloaded_data_split.train, splitter_name
    )
    assert_frames_are_identical(
        initial_data_split.val, reloaded_data_split.val, splitter_name
    )
    assert_frames_are_identical(
        initial_data_split.test, reloaded_data_split.test, splitter_name
    )


def test_save_path_not_provided_no_saving(sample_dataframe, tmp_path):
    """Test that no file is saved when save_path is not provided."""
    splitter = RatioSplitter(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
    data_split = splitter(sample_dataframe)
    
    # Verify no files were created in tmp_path
    assert len(list(tmp_path.iterdir())) == 0
    
    # Verify the split still works correctly
    assert isinstance(data_split, DataSplit)
    assert not data_split.train.empty
    assert not data_split.val.empty
    assert not data_split.test.empty


def test_save_path_creates_directory_structure(sample_dataframe, tmp_path):
    """Test that directory structure is created when it doesn't exist."""
    nested_path = tmp_path / "deep" / "nested" / "path" / "indices.pkl"
    
    splitter = RatioSplitter(
        train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, save_path=str(nested_path)
    )
    data_split = splitter(sample_dataframe)
    
    # Verify the file was created and directory structure exists
    assert nested_path.exists()
    assert nested_path.is_file()
    
    # Verify the parent directories were created
    assert nested_path.parent.exists()
    assert nested_path.parent.is_dir()


def test_saved_indices_format_and_content(sample_dataframe, tmp_path):
    """Test that saved indices have the correct format and content."""
    save_path = tmp_path / "test_indices.pkl"
    
    splitter = RatioSplitter(
        train_ratio=0.6, val_ratio=0.3, test_ratio=0.1, save_path=str(save_path)
    )
    data_split = splitter(sample_dataframe)
    
    # Load the saved indices
    with open(save_path, "rb") as f:
        saved_data = pickle.load(f)
    
    # Verify the structure: should be a list containing one dict
    assert isinstance(saved_data, list)
    assert len(saved_data) == 1
    
    indices_dict = saved_data[0]
    assert isinstance(indices_dict, dict)
    assert set(indices_dict.keys()) == {"train", "val", "test"}
    
    # Verify indices are lists of integers
    for split_name, indices in indices_dict.items():
        assert isinstance(indices, list)
        assert all(isinstance(idx, int) for idx in indices)
    
    # Verify no overlapping indices
    all_indices = indices_dict["train"] + indices_dict["val"] + indices_dict["test"]
    assert len(all_indices) == len(set(all_indices)), "Found duplicate indices across splits"
    
    # Verify total count matches DataFrame length
    assert len(all_indices) == len(sample_dataframe)
    
    # Verify indices are valid (within DataFrame range)
    assert all(0 <= idx < len(sample_dataframe) for idx in all_indices)


def test_saved_indices_match_actual_splits(sample_dataframe, tmp_path):
    """Test that saved indices exactly match the indices of the actual splits."""
    save_path = tmp_path / "match_test.pkl"
    
    splitter = RatioSplitter(
        train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, save_path=str(save_path)
    )
    data_split = splitter(sample_dataframe)
    
    # Load the saved indices
    with open(save_path, "rb") as f:
        saved_data = pickle.load(f)
    
    indices_dict = saved_data[0]
    
    # Verify that the saved indices match the actual DataFrame indices
    assert sorted(data_split.train.index.tolist()) == sorted(indices_dict["train"])
    assert sorted(data_split.val.index.tolist()) == sorted(indices_dict["val"])
    assert sorted(data_split.test.index.tolist()) == sorted(indices_dict["test"])


def test_save_path_invalid_extension_raises_error(sample_dataframe):
    """Test that invalid save_path extension raises ValueError."""
    with pytest.raises(ValueError, match="save_path must end with '.pkl' if provided"):
        RatioSplitter(
            train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, 
            save_path="invalid_path.txt"
        )
    
    with pytest.raises(ValueError, match="save_path must end with '.pkl' if provided"):
        RatioSplitter(
            train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, 
            save_path="no_extension"
        )


def test_save_path_overwrite_existing_file(sample_dataframe, tmp_path):
    """Test that existing files are overwritten when save_path is reused."""
    save_path = tmp_path / "overwrite_test.pkl"
    
    # Create initial file with different content
    with open(save_path, "wb") as f:
        pickle.dump(["initial", "content"], f)
    
    assert save_path.exists()
    initial_size = save_path.stat().st_size
    
    # Run splitter to overwrite the file
    splitter = RatioSplitter(
        train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, save_path=str(save_path)
    )
    data_split = splitter(sample_dataframe)
    
    # Verify file was overwritten with correct content
    assert save_path.exists()
    
    with open(save_path, "rb") as f:
        saved_data = pickle.load(f)
    
    # Should now contain the split indices, not the initial content
    assert isinstance(saved_data, list)
    assert len(saved_data) == 1
    assert isinstance(saved_data[0], dict)
    assert set(saved_data[0].keys()) == {"train", "val", "test"}


@pytest.mark.parametrize("splitter_class,splitter_args,fixture_name", [
    (RatioSplitter, {"train_ratio": 0.7, "val_ratio": 0.2, "test_ratio": 0.1}, "sample_dataframe"),
    (SizeSplitter, {"train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1}, "size_splitter_dataframe"),
    (TargetSplitter, {"train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1}, "target_splitter_dataframe"),
    (ScaffoldSplitter, {"train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1}, "scaffold_dataframe"),
])
def test_all_splitters_save_indices_correctly(splitter_class, splitter_args, fixture_name, request, tmp_path):
    """Test that all splitter types save indices correctly."""
    dataframe = request.getfixturevalue(fixture_name)
    save_path = tmp_path / f"{splitter_class.__name__}_test.pkl"
    
    # Add save_path to args
    args_with_save = splitter_args.copy()
    args_with_save["save_path"] = str(save_path)
    
    splitter = splitter_class(**args_with_save)
    data_split = splitter(dataframe)
    
    # Verify file exists
    assert save_path.exists()
    
    # Verify content structure
    with open(save_path, "rb") as f:
        saved_data = pickle.load(f)
    
    assert isinstance(saved_data, list)
    assert len(saved_data) == 1
    indices_dict = saved_data[0]
    assert set(indices_dict.keys()) == {"train", "val", "test"}
    
    # Verify the indices can be used to reconstruct the splits
    reconstructed_train = dataframe.iloc[indices_dict["train"]]
    reconstructed_val = dataframe.iloc[indices_dict["val"]]
    reconstructed_test = dataframe.iloc[indices_dict["test"]]
    
    # Compare content (reset index for comparison since order might differ)
    pd.testing.assert_frame_equal(
        data_split.train.reset_index(drop=True).sort_index(axis=1),
        reconstructed_train.reset_index(drop=True).sort_index(axis=1)
    )
    pd.testing.assert_frame_equal(
        data_split.val.reset_index(drop=True).sort_index(axis=1),
        reconstructed_val.reset_index(drop=True).sort_index(axis=1)
    )
    pd.testing.assert_frame_equal(
        data_split.test.reset_index(drop=True).sort_index(axis=1),
        reconstructed_test.reset_index(drop=True).sort_index(axis=1)
    )


def test_empty_splits_handling(tmp_path):
    """Test handling of edge cases with very small datasets that might create empty splits."""
    # Create a tiny dataframe that might result in empty splits
    tiny_df = pd.DataFrame({"col1": [1, 2], "col2": [10, 20]})
    save_path = tmp_path / "tiny_test.pkl"
    
    # Use ratios that would normally create empty splits
    splitter = RatioSplitter(
        train_ratio=0.9, val_ratio=0.05, test_ratio=0.05, save_path=str(save_path)
    )
    
    data_split = splitter(tiny_df)
    
    # Verify file was created
    assert save_path.exists()
    
    # Load and verify indices
    with open(save_path, "rb") as f:
        saved_data = pickle.load(f)
    
    indices_dict = saved_data[0]
    
    # Verify all indices are valid even if some splits might be empty
    for split_name, indices in indices_dict.items():
        assert isinstance(indices, list)
        for idx in indices:
            assert 0 <= idx < len(tiny_df)
    
    # Verify total indices count
    total_indices = len(indices_dict["train"]) + len(indices_dict["val"]) + len(indices_dict["test"])
    assert total_indices == len(tiny_df)


def test_empty_validation_split_behavior(sample_dataframe, tmp_path):
    """Test that empty validation split (0.9, 0.0, 0.1) works correctly."""
    save_path = tmp_path / "empty_val_split.pkl"
    
    splitter = RatioSplitter(
        train_ratio=0.9, val_ratio=0.0, test_ratio=0.1, save_path=str(save_path)
    )
    data_split = splitter(sample_dataframe)
    
    # Verify split behavior
    assert not data_split.train.empty, "Train split should not be empty"
    assert data_split.val.empty, "Validation split should be empty"
    assert not data_split.test.empty, "Test split should not be empty"
    
    # Verify split sizes
    expected_train_size = int(len(sample_dataframe) * 0.9)
    expected_test_size = len(sample_dataframe) - expected_train_size
    
    assert len(data_split.train) == expected_train_size
    assert len(data_split.val) == 0
    assert len(data_split.test) == expected_test_size
    
    # Verify empty DataFrame properties
    assert data_split.val.shape == (0, len(sample_dataframe.columns))
    assert list(data_split.val.columns) == list(sample_dataframe.columns)
    
    # Verify saved indices
    with open(save_path, "rb") as f:
        saved_data = pickle.load(f)
    
    indices_dict = saved_data[0]
    assert len(indices_dict["train"]) == expected_train_size
    assert len(indices_dict["val"]) == 0
    assert len(indices_dict["test"]) == expected_test_size
    assert indices_dict["val"] == []  # Empty list specifically


def test_only_train_split_behavior(sample_dataframe, tmp_path):
    """Test that only train split (1.0, 0.0, 0.0) works correctly."""
    save_path = tmp_path / "only_train_split.pkl"
    
    splitter = RatioSplitter(
        train_ratio=1.0, val_ratio=0.0, test_ratio=0.0, save_path=str(save_path)
    )
    data_split = splitter(sample_dataframe)
    
    # Verify split behavior
    assert not data_split.train.empty, "Train split should not be empty"
    assert data_split.val.empty, "Validation split should be empty"
    assert data_split.test.empty, "Test split should be empty"
    
    # Verify all data goes to train
    assert len(data_split.train) == len(sample_dataframe)
    assert len(data_split.val) == 0
    assert len(data_split.test) == 0
    
    # Verify empty DataFrames maintain structure
    for empty_split in [data_split.val, data_split.test]:
        assert empty_split.shape == (0, len(sample_dataframe.columns))
        assert list(empty_split.columns) == list(sample_dataframe.columns)
    
    # Verify saved indices
    with open(save_path, "rb") as f:
        saved_data = pickle.load(f)
    
    indices_dict = saved_data[0]
    assert len(indices_dict["train"]) == len(sample_dataframe)
    assert len(indices_dict["val"]) == 0
    assert len(indices_dict["test"]) == 0
    assert indices_dict["val"] == []
    assert indices_dict["test"] == []


def test_index_splitter_loads_empty_splits(sample_dataframe, tmp_path):
    """Test that IndexSplitter correctly loads and handles empty splits from pickle files."""
    save_path = tmp_path / "empty_splits_for_loading.pkl"
    
    # First create a split with empty validation using RatioSplitter
    ratio_splitter = RatioSplitter(
        train_ratio=0.8, val_ratio=0.0, test_ratio=0.2, save_path=str(save_path)
    )
    original_split = ratio_splitter(sample_dataframe)
    
    # Now load it with IndexSplitter
    index_splitter = IndexSplitter(split_index_path=str(save_path))
    loaded_split = index_splitter(sample_dataframe)
    
    # Verify the loaded split matches the original
    assert len(loaded_split.train) == len(original_split.train)
    assert len(loaded_split.val) == len(original_split.val) == 0
    assert len(loaded_split.test) == len(original_split.test)
    
    # Verify empty validation split properties
    assert loaded_split.val.empty
    assert loaded_split.val.shape == (0, len(sample_dataframe.columns))
    assert list(loaded_split.val.columns) == list(sample_dataframe.columns)
    
    # Verify indices match exactly
    assert sorted(loaded_split.train.index.tolist()) == sorted(original_split.train.index.tolist())
    assert loaded_split.val.index.tolist() == []
    assert sorted(loaded_split.test.index.tolist()) == sorted(original_split.test.index.tolist())


def test_index_splitter_loads_only_train_split(sample_dataframe, tmp_path):
    """Test IndexSplitter with a split containing only training data."""
    save_path = tmp_path / "only_train_for_loading.pkl"
    
    # Create a split with only train data
    ratio_splitter = RatioSplitter(
        train_ratio=1.0, val_ratio=0.0, test_ratio=0.0, save_path=str(save_path)
    )
    original_split = ratio_splitter(sample_dataframe)
    
    # Load with IndexSplitter
    index_splitter = IndexSplitter(split_index_path=str(save_path))
    loaded_split = index_splitter(sample_dataframe)
    
    # Verify all data in train, empty val and test
    assert len(loaded_split.train) == len(sample_dataframe)
    assert len(loaded_split.val) == 0
    assert len(loaded_split.test) == 0
    
    # Verify empty splits maintain DataFrame structure
    for empty_split in [loaded_split.val, loaded_split.test]:
        assert empty_split.empty
        assert empty_split.shape == (0, len(sample_dataframe.columns))
        assert list(empty_split.columns) == list(sample_dataframe.columns)
    
    # Verify train contains all original data
    pd.testing.assert_frame_equal(
        loaded_split.train.sort_index(),
        sample_dataframe.sort_index()
    )


def test_index_splitter_handles_manually_created_empty_indices(sample_dataframe, tmp_path):
    """Test IndexSplitter with manually created pickle files containing empty index lists."""
    save_path = tmp_path / "manual_empty_indices.pkl"
    
    # Manually create a pickle file with empty validation indices
    manual_indices = {
        "train": list(range(8)),  # First 8 indices
        "val": [],                # Empty validation
        "test": [8, 9]           # Last 2 indices
    }
    
    with open(save_path, "wb") as f:
        pickle.dump([manual_indices], f)
    
    # Load with IndexSplitter
    index_splitter = IndexSplitter(split_index_path=str(save_path))
    loaded_split = index_splitter(sample_dataframe)
    
    # Verify correct loading
    assert len(loaded_split.train) == 8
    assert len(loaded_split.val) == 0
    assert len(loaded_split.test) == 2
    
    # Verify empty validation properties
    assert loaded_split.val.empty
    assert loaded_split.val.shape == (0, len(sample_dataframe.columns))
    
    # Verify correct indices were loaded
    assert sorted(loaded_split.train.index.tolist()) == list(range(8))
    assert loaded_split.val.index.tolist() == []
    assert sorted(loaded_split.test.index.tolist()) == [8, 9]


@pytest.mark.parametrize("empty_split_ratios,expected_empty_splits", [
    ((0.8, 0.0, 0.2), ["val"]),
    ((0.7, 0.3, 0.0), ["test"]),
    ((1.0, 0.0, 0.0), ["val", "test"]),
    ((0.0, 0.5, 0.5), ["train"]),
    ((0.0, 1.0, 0.0), ["train", "test"]),
    ((0.0, 0.0, 1.0), ["train", "val"]),
])
def test_empty_splits_parametrized(sample_dataframe, tmp_path, empty_split_ratios, expected_empty_splits):
    """Parametrized test for various empty split combinations."""
    train_ratio, val_ratio, test_ratio = empty_split_ratios
    save_path = tmp_path / f"empty_splits_{train_ratio}_{val_ratio}_{test_ratio}.pkl"
    
    splitter = RatioSplitter(
        train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, 
        save_path=str(save_path)
    )
    data_split = splitter(sample_dataframe)
    
    # Check which splits should be empty
    for split_name in ["train", "val", "test"]:
        split_data = getattr(data_split, split_name)
        if split_name in expected_empty_splits:
            assert split_data.empty, f"{split_name} split should be empty"
            assert len(split_data) == 0
            assert split_data.shape == (0, len(sample_dataframe.columns))
        else:
            assert not split_data.empty, f"{split_name} split should not be empty"
            assert len(split_data) > 0
    
    # Verify total length is preserved
    total_length = len(data_split.train) + len(data_split.val) + len(data_split.test)
    assert total_length == len(sample_dataframe)
    
    # Test IndexSplitter can load these splits
    index_splitter = IndexSplitter(split_index_path=str(save_path))
    loaded_split = index_splitter(sample_dataframe)
    
    # Verify loaded splits match original
    for split_name in ["train", "val", "test"]:
        original = getattr(data_split, split_name)
        loaded = getattr(loaded_split, split_name)
        assert len(original) == len(loaded)
        if not original.empty:
            pd.testing.assert_frame_equal(
                original.sort_index(),
                loaded.sort_index()
            )
