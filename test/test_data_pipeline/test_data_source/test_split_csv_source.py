import pandas as pd
import pytest

from chemtorch.data_pipeline.data_source import PreSplitCSVSource
from chemtorch.utils import DataSplit


@pytest.fixture
def split_csv_folder(tmp_path):
    """Fixture to create a temporary folder with train/val/test CSV files for testing SplitCSVSource."""
    folder_path = tmp_path / "data"
    folder_path.mkdir()
    for split in ["train", "val", "test"]:
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        df.to_csv(folder_path / f"{split}.csv", index=False)
    return str(folder_path)


def test_split_csv_source(split_csv_folder):
    """Test instantiation and forward pass of SplitCSVSource."""
    reader = PreSplitCSVSource(data_folder=split_csv_folder)
    data_split = reader.load()
    assert isinstance(data_split, DataSplit)
    assert isinstance(data_split.train, pd.DataFrame)
    assert isinstance(data_split.val, pd.DataFrame)
    assert isinstance(data_split.test, pd.DataFrame)
    assert not data_split.train.empty
    assert not data_split.val.empty
    assert not data_split.test.empty


def test_split_csv_source_missing_files(tmp_path):
    """Test SplitCSVSource with missing CSV files."""
    data_folder = tmp_path / "data"
    data_folder.mkdir()
    # Create only one file instead of all three
    (data_folder / "train.csv").write_text("col1,col2\n1,2\n3,4\n")

    reader = PreSplitCSVSource(data_folder=str(data_folder))
    with pytest.raises(FileNotFoundError, match="Missing files"):
        reader.load()


def test_split_csv_source_empty_files(tmp_path):
    """Test SplitCSVSource with empty CSV files."""
    data_folder = tmp_path / "data"
    data_folder.mkdir()
    for split in ["train", "val", "test"]:
        (data_folder / f"{split}.csv").write_text("")  # Create empty files

    reader = PreSplitCSVSource(data_folder=str(data_folder))
    with pytest.raises(pd.errors.EmptyDataError, match="No columns to parse from file"):
        reader.load()
