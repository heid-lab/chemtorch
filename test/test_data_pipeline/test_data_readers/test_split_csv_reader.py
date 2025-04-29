import pandas as pd
import pytest
from deeprxn.data import DataSplit
from deeprxn.data_reader.split_csv_reader import SplitCSVReader


@pytest.fixture
def split_csv_folder(tmp_path):
    """Fixture to create a temporary folder with train/val/test CSV files for testing SplitCSVReader."""
    folder_path = tmp_path / "data"
    folder_path.mkdir()
    for split in ["train", "val", "test"]:
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        df.to_csv(folder_path / f"{split}.csv", index=False)
    return str(folder_path)


def test_split_csv_reader(split_csv_folder):
    """Test instantiation and forward pass of SplitCSVReader."""
    reader = SplitCSVReader(data_folder=split_csv_folder)
    data_split = reader.forward()
    assert isinstance(data_split, DataSplit)
    assert isinstance(data_split.train, pd.DataFrame)
    assert isinstance(data_split.val, pd.DataFrame)
    assert isinstance(data_split.test, pd.DataFrame)
    assert not data_split.train.empty
    assert not data_split.val.empty
    assert not data_split.test.empty