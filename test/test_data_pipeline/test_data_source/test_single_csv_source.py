import pytest
import pandas as pd
from deeprxn.data_pipeline.data_source.single_csv_source import SingleCSVSource

@pytest.fixture
def single_csv_file(tmp_path):
    """Fixture to create a temporary CSV file for testing SingleCSVSource."""
    file_path = tmp_path / "data.csv"
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    df.to_csv(file_path, index=False)
    return str(file_path)

def test_single_csv_source(single_csv_file):
    """Test instantiation and forward pass of SingleCSVSource."""
    reader = SingleCSVSource(data_path=single_csv_file)
    df = reader.load()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert list(df.columns) == ["col1", "col2"]
