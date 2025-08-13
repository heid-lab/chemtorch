# conftest.py
import pickle
import pandas as pd
import pytest


@pytest.fixture
def sample_dataframe():
    """Fixture to create a sample DataFrame for testing basic splitters."""
    return pd.DataFrame({"col1": range(1, 11), "col2": range(11, 21), "id": range(10)})


@pytest.fixture
def index_pickle_file(tmp_path):
    """Fixture to create a temporary pickle file with train, val, and test indices."""
    split_index_path = tmp_path / "indices.pkl"
    indices = [[0, 1, 2, 3, 4], [5, 6], [7, 8, 9]]  # Train, val, test indices
    with open(split_index_path, "wb") as f:
        pickle.dump([indices], f)
    return str(split_index_path)


@pytest.fixture
def scaffold_dataframe():
    """Fixture to create a sample DataFrame for testing ScaffoldSplitter."""
    data = {
        "smiles": [
            "c1ccccc1.O>>CCO",
            "c1ccccc1.N>>CCN",
            "c1ccncc1>>CCNc1ccncc1",
            "C1C[C@H]2C[C@@H]1C2>>product",
            "C1CC2CC1C2>>product",
            "CC>>CCO",
            "CCC>>CCCO",
            "C1CC1.c1ccccc1>>C1CC1c1ccccc1",
            "invalid>>smiles",
        ],
        "id": range(9),
    }
    return pd.DataFrame(data)


@pytest.fixture
def size_splitter_dataframe():
    """Fixture for testing SizeSplitter with varying molecule sizes."""
    smiles_data = [f"{'C' * i}>>{'C' * (i + 1)}" for i in range(1, 21)]  # 20 samples
    return pd.DataFrame({"smiles": smiles_data, "id": range(20)})


@pytest.fixture
def target_splitter_dataframe():
    """Fixture for testing TargetSplitter with a numeric label progression."""
    return pd.DataFrame({"label": range(20), "id": range(20)})
