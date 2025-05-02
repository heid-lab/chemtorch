import pytest
import pandas as pd
from deeprxn.data_pipeline.data_reader.single_csv_reader import SingleCSVReader

@pytest.fixture
def single_csv_file(tmp_path):
    """Fixture to create a temporary CSV file for testing SingleCSVReader."""
    file_path = tmp_path / "data.csv"
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    df.to_csv(file_path, index=False)
    return str(file_path)

def test_single_csv_reader(single_csv_file):
    """Test instantiation and forward pass of SingleCSVReader."""
    reader = SingleCSVReader(data_path=single_csv_file)
    df = reader.forward()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert list(df.columns) == ["col1", "col2"]
