import pandas as pd
import pytest
from deepreaction.data_pipeline.data_split import DataSplit
from deepreaction.data_pipeline.column_mapper.column_filter_rename import ColumnFilterAndRename

def test_column_filter_and_rename_success():
    # Mock input DataFrames
    train_df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
    val_df = pd.DataFrame({"A": [7, 8], "B": [9, 10], "C": [11, 12]})
    test_df = pd.DataFrame({"A": [13, 14], "B": [15, 16], "C": [17, 18]})
    data_split = DataSplit(train_df, val_df, test_df)

    # Column mapping
    column_mapping = {"X": "A", "Y": "B"}

    # Instantiate and apply the column mapper
    column_mapper = ColumnFilterAndRename(column_mapping)
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

    # Column mapping with a missing column
    column_mapping = {"X": "A", "Y": "C"}  # "C" is missing

    # Instantiate the column mapper
    column_mapper = ColumnFilterAndRename(column_mapping)

    # Assert that a KeyError is raised
    with pytest.raises(KeyError):
        column_mapper(data_split)

def test_column_filter_and_rename_invalid_input():
    # Invalid input (not a DataSplit object)
    invalid_input = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

    # Column mapping
    column_mapping = {"X": "A", "Y": "B"}

    # Instantiate the column mapper
    column_mapper = ColumnFilterAndRename(column_mapping)

    # Assert that a TypeError is raised
    with pytest.raises(TypeError):
        column_mapper(invalid_input)