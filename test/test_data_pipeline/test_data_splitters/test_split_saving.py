import pandas as pd
import pytest

from chemtorch.components.data_pipeline.data_splitter import (
    IndexSplitter,
    RatioSplitter,
    ScaffoldSplitter,
    SizeSplitter,
    TargetSplitter,
)


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
    current_args.update(
        {"save_split_dir": str(save_dir), "save_indices": True, "save_csv": True}
    )
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

    # --- 4. Verification: Saved CSV files ---
    train_csv_path = save_dir / "train.csv"
    assert train_csv_path.exists(), f"train.csv not found for {splitter_name}"

    # Load CSVs back into memory
    loaded_train_csv = pd.read_csv(train_csv_path)
    loaded_val_csv = pd.read_csv(save_dir / "val.csv")
    loaded_test_csv = pd.read_csv(save_dir / "test.csv")

    # To compare the initial split with the loaded CSV, we must account for
    # the fact that the CSV has a new 0-based index. We do this by resetting
    # the index of the initial split's DataFrames just for this comparison.
    pd.testing.assert_frame_equal(
        initial_data_split.train.reset_index(drop=True), loaded_train_csv
    )
    pd.testing.assert_frame_equal(
        initial_data_split.val.reset_index(drop=True), loaded_val_csv
    )
    pd.testing.assert_frame_equal(
        initial_data_split.test.reset_index(drop=True), loaded_test_csv
    )
