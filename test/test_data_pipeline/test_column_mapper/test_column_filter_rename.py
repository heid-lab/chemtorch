import pandas as pd
import pytest

from chemtorch.components.data_pipeline.column_mapper import ColumnFilterAndRename
from chemtorch.utils import DataSplit


def test_column_filter_and_rename_success():
    # Mock input DataFrames
    train_df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
    val_df = pd.DataFrame({"A": [7, 8], "B": [9, 10], "C": [11, 12]})
    test_df = pd.DataFrame({"A": [13, 14], "B": [15, 16], "C": [17, 18]})
    data_split = DataSplit(train_df, val_df, test_df)

    # Instantiate and apply the column mapper using kwargs
    column_mapper = ColumnFilterAndRename(X="A", Y="B")
    processed_data_split = column_mapper(data_split)

    # Assert the columns are renamed and filtered correctly
    for df in processed_data_split:
        assert list(df.columns) == ["X", "Y"]


def test_column_filter_and_rename_missing_columns():
    # Mock input DataFrames
    train_df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    val_df = pd.DataFrame({"A": [7, 8], "B": [9, 10]})
    test_df = pd.DataFrame({"A": [13, 14], "B": [15, 16]})
    data_split = DataSplit(train_df, val_df, test_df)

    # Instantiate the column mapper with a missing column using kwargs
    column_mapper = ColumnFilterAndRename(X="A", Y="C")  # "C" is missing

    # Assert that a KeyError is raised
    with pytest.raises(KeyError):
        column_mapper(data_split)


def test_column_filter_and_rename_invalid_input():
    # Invalid input (not a DataSplit object or a pandas DataFrame)
    invalid_input = {"A": [1, 2], "B": [3, 4]}

    # Instantiate the column mapper using kwargs
    column_mapper = ColumnFilterAndRename(X="A", Y="B")

    # Assert that a TypeError is raised
    with pytest.raises(TypeError):
        column_mapper(invalid_input)


def test_column_filter_and_rename_single_dataframe_success():
    # Mock input DataFrame
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
    column_mapper = ColumnFilterAndRename(X="A", Y="B")
    processed_df = column_mapper(df)
    assert isinstance(processed_df, pd.DataFrame)
    assert list(processed_df.columns) == ["X", "Y"]
    assert processed_df["X"].tolist() == [1, 2]
    assert processed_df["Y"].tolist() == [3, 4]


def test_column_filter_and_rename_single_dataframe_missing_column():
    # Mock input DataFrame
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    column_mapper = ColumnFilterAndRename(X="A", Y="C")  # "C" is missing
    with pytest.raises(KeyError):
        column_mapper(df)


def test_column_filter_and_rename_single_dataframe_invalid_input():
    # Invalid input (not a DataFrame)
    invalid_input = [1, 2, 3]
    column_mapper = ColumnFilterAndRename(X="A")
    with pytest.raises(TypeError):
        column_mapper(invalid_input)


def test_column_filter_and_rename_no_mappings():
    # Test that ValueError is raised when no column mappings are provided
    with pytest.raises(ValueError, match="At least one column mapping must be provided"):
        ColumnFilterAndRename()


def test_column_filter_and_rename_none_values_filtered():
    # Test that None values are filtered out
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
    column_mapper = ColumnFilterAndRename(X="A", Y="B", Z=None)  # Z=None should be filtered out
    processed_df = column_mapper(df)
    assert isinstance(processed_df, pd.DataFrame)
    assert list(processed_df.columns) == ["X", "Y"]  # Z should not be present
