import pytest
import pandas as pd
from torch import nn

from deepreaction.data_pipeline.column_mapper.column_filter_rename import ColumnFilterAndRename
from deepreaction.data_pipeline.data_source.data_source import DataSource
from deepreaction.data_pipeline.data_splitter.data_splitter import DataSplitter
from deepreaction.data_pipeline.data_split import DataSplit
from deepreaction.data_pipeline.data_source.single_csv_source import SingleCSVSource
from deepreaction.data_pipeline.data_source.split_csv_source import SplitCSVSource
from deepreaction.data_pipeline.data_splitter.ratio_splitter import RatioSplitter

class NoOpMockSource(DataSource):
    def load(self):
        return None  # Return invalid data

class NoOpMockSplitter(DataSplitter):
    def forward(self, raw):
        return None  # Return invalid data

@pytest.fixture
def single_csv_file(tmp_path):
    """Fixture to create a temporary CSV file for testing SingleCSVSource."""
    file_path = tmp_path / "data.csv"
    df = pd.DataFrame({
        "col1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
        "col2": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        })
    df.to_csv(file_path, index=False)
    return str(file_path)

@pytest.fixture
def split_csv_folder(tmp_path):
    """Fixture to create a temporary folder with train/val/test CSV files for testing SplitCSVSource."""
    folder_path = tmp_path / "data"
    folder_path.mkdir()
    for split in ["train", "val", "test"]:
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        df.to_csv(folder_path / f"{split}.csv", index=False)
    return str(folder_path)

def test_preprocessing_with_single_csv_source(single_csv_file):
    """Test the data pipeline with SingleCSVSource and RatioSplitter."""
    source = SingleCSVSource(data_path=single_csv_file)
    splitter = RatioSplitter(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    data = source.load()
    pipeline = nn.Sequential(splitter)
    data_split = pipeline.forward(data)
    assert isinstance(data_split, DataSplit)
    assert isinstance(data_split.train, pd.DataFrame)
    assert isinstance(data_split.val, pd.DataFrame)
    assert isinstance(data_split.test, pd.DataFrame)
    assert not data_split.train.empty
    assert not data_split.val.empty
    assert not data_split.test.empty

def test_preprocessing_with_split_csv_source(split_csv_folder):
    """Test the data pipeline with SplitCSVSource."""
    source = SplitCSVSource(data_folder=split_csv_folder)
    pipeline = nn.Sequential()
    data = source.load()
    data_split = pipeline.forward(data)
    assert isinstance(data_split, DataSplit)
    assert isinstance(data_split.train, pd.DataFrame)
    assert isinstance(data_split.val, pd.DataFrame)
    assert isinstance(data_split.test, pd.DataFrame)
    assert not data_split.train.empty
    assert not data_split.val.empty
    assert not data_split.test.empty


def test_preprocessing_with_single_csv_source_and_column_mapper(single_csv_file):
    """Test the data pipeline with SingleCSVSource, RatioSplitter, and ColumnFilterAndRename."""
    source = SingleCSVSource(data_path=single_csv_file)
    splitter = RatioSplitter(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    column_mapper = ColumnFilterAndRename(column_mapping={"new_col1": "col1", "new_col2": "col2"})

    # Load data and process through the pipeline
    data = source.load()
    pipeline = nn.Sequential(splitter, column_mapper)
    data_split = pipeline.forward(data)

    # Assertions
    assert isinstance(data_split, DataSplit)
    for df in [data_split.train, data_split.val, data_split.test]:
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert list(df.columns) == ["new_col1", "new_col2"]

def test_preprocessing_with_split_csv_source_and_column_mapper(split_csv_folder):
    """Test the data pipeline with SplitCSVSource and ColumnFilterAndRename."""
    source = SplitCSVSource(data_folder=split_csv_folder)
    column_mapper = ColumnFilterAndRename(column_mapping={"new_col1": "col1", "new_col2": "col2"})

    # Load data and process through the pipeline
    data = source.load()
    pipeline = nn.Sequential(column_mapper)
    data_split = pipeline.forward(data)

    # Assertions
    assert isinstance(data_split, DataSplit)
    for df in [data_split.train, data_split.val, data_split.test]:
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert list(df.columns) == ["new_col1", "new_col2"]
