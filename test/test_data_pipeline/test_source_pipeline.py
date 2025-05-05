import pytest
import pandas as pd
from deeprxn.data_pipeline.data_reader.data_reader import DataReader
from deeprxn.data_pipeline.data_splitter.data_splitter import DataSplitter
from deeprxn.data_pipeline.data_pipeline import DataSourcePipeline, DataSplit
from deeprxn.data_pipeline.data_reader.single_csv_reader import SingleCSVReader
from deeprxn.data_pipeline.data_reader.split_csv_reader import SplitCSVReader
from deeprxn.data_pipeline.data_splitter.ratio_splitter import RatioSplitter

class NoOpMockReader(DataReader):
    def forward(self):
        return None  # Return invalid data

class NoOpMockSplitter(DataSplitter):
    def forward(self, raw):
        return None  # Return invalid data

@pytest.fixture
def single_csv_file(tmp_path):
    """Fixture to create a temporary CSV file for testing SingleCSVReader."""
    file_path = tmp_path / "data.csv"
    df = pd.DataFrame({
        "col1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
        "col2": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        })
    df.to_csv(file_path, index=False)
    return str(file_path)

@pytest.fixture
def split_csv_folder(tmp_path):
    """Fixture to create a temporary folder with train/val/test CSV files for testing SplitCSVReader."""
    folder_path = tmp_path / "data"
    folder_path.mkdir()
    for split in ["train", "val", "test"]:
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        df.to_csv(folder_path / f"{split}.csv", index=False)
    return str(folder_path)

def test_source_pipeline_with_single_csv_reader(single_csv_file):
    """Test the data pipeline with SingleCSVReader and RatioSplitter."""
    reader = SingleCSVReader(data_path=single_csv_file)
    splitter = RatioSplitter(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    pipeline = DataSourcePipeline(components=[reader, splitter])
    data_split = pipeline.forward()
    assert isinstance(data_split, DataSplit)
    assert isinstance(data_split.train, pd.DataFrame)
    assert isinstance(data_split.val, pd.DataFrame)
    assert isinstance(data_split.test, pd.DataFrame)
    assert not data_split.train.empty
    assert not data_split.val.empty
    assert not data_split.test.empty

def test_source_pipeline_with_split_csv_reader(split_csv_folder):
    """Test the data pipeline with SplitCSVReader."""
    reader = SplitCSVReader(data_folder=split_csv_folder)
    pipeline = DataSourcePipeline(components=[reader])
    data_split = pipeline.forward()
    assert isinstance(data_split, DataSplit)
    assert isinstance(data_split.train, pd.DataFrame)
    assert isinstance(data_split.val, pd.DataFrame)
    assert isinstance(data_split.test, pd.DataFrame)
    assert not data_split.train.empty
    assert not data_split.val.empty
    assert not data_split.test.empty


def test_source_pipeline_invalid_reader():
    """Test DataPipeline with an invalid reader."""
    reader = NoOpMockReader()
    splitter = NoOpMockSplitter()
    pipeline = DataSourcePipeline(components=[reader, splitter])
    with pytest.raises(TypeError, match="Final output must be a DataSplit object"):
        pipeline.forward()