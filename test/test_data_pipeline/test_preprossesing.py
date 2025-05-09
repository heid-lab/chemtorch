import pytest
import pandas as pd
from deeprxn.data_pipeline.data_source.data_source import DataSource
from deeprxn.data_pipeline.data_splitter.data_splitter import DataSplitter
from deeprxn.data_pipeline.data_pipeline import DataPipeline, DataSplit
from deeprxn.data_pipeline.data_source.single_csv_source import SingleCSVSource
from deeprxn.data_pipeline.data_source.split_csv_source import SplitCSVSource
from deeprxn.data_pipeline.data_splitter.ratio_splitter import RatioSplitter

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
    pipeline = DataPipeline(components=[splitter])
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
    pipeline = DataPipeline([])
    data = source.load()
    data_split = pipeline.forward(data)
    assert isinstance(data_split, DataSplit)
    assert isinstance(data_split.train, pd.DataFrame)
    assert isinstance(data_split.val, pd.DataFrame)
    assert isinstance(data_split.test, pd.DataFrame)
    assert not data_split.train.empty
    assert not data_split.val.empty
    assert not data_split.test.empty
