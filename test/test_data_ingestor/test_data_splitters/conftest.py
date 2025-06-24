import pandas as pd
import pytest


@pytest.fixture
def sample_dataframe():
    """Fixture to create a sample DataFrame for testing splitters."""
    return pd.DataFrame({"col1": range(1, 11), "col2": range(11, 21)})
