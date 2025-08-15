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
    # The IndexSplitter expects a pickle file containing a tuple/list, where the first element is a list of 3 arrays
    indices = [[0, 1, 2, 3, 4], [5, 6], [7, 8, 9]]  # Train, val, test indices
    with open(split_index_path, "wb") as f:
        pickle.dump([indices], f)
    return str(split_index_path)


@pytest.fixture
def invalid_index_pickle_file(tmp_path):
    """Fixture to create an invalid pickle file with incorrect indices."""
    split_index_path = tmp_path / "invalid_indices.pkl"
    indices = [[0, 1, 2], [3, 4]]  # Only 2 splits instead of 3
    with open(split_index_path, "wb") as f:
        pickle.dump([indices], f)
    return str(split_index_path)


@pytest.fixture
def malformed_pickle_file(tmp_path):
    """Fixture to create a malformed pickle file (not a list/tuple)."""
    split_index_path = tmp_path / "malformed.pkl"
    with open(split_index_path, "wb") as f:
        pickle.dump("not a list", f)
    return str(split_index_path)


@pytest.fixture
def single_molecule_dataframe():
    """
    Fixture to create a sample DataFrame for testing ScaffoldSplitter with single molecules.
    """
    data = {
        "smiles": [
            "c1ccccc1",  # Benzene scaffold
            "c1ccncc1",  # Pyridine scaffold
            "C1C[C@H]2C[C@@H]1C2",  # Chiral norbornane
            "C1CC2CC1C2",  # Non-chiral norbornane
            "CC",  # Acyclic, no scaffold
            "CCC",  # Acyclic, no scaffold
            "c1ccc2ccccc2c1",  # Naphthalene scaffold
            "C1CCCCC1",  # Cyclohexane scaffold
        ],
        "id": range(8),
    }
    return pd.DataFrame(data)


@pytest.fixture
def scaffold_dataframe():
    """
    Fixture to create a sample DataFrame for testing ScaffoldSplitter.
    Includes molecules with shared scaffolds, different scaffolds, no scaffolds,
    and chirality to test the include_chirality flag.
    """
    data = {
        "smiles": [
            "c1ccccc1.O>>CCO",  # Benzene scaffold, idx 0
            "c1ccccc1.N>>CCN",  # Benzene scaffold, idx 1
            "c1ccncc1>>CCNc1ccncc1",  # Pyridine scaffold, idx 2
            # Norbornane examples for chirality test
            "C1C[C@H]2C[C@@H]1C2>>product",  # Chiral molecule, idx 3
            "C1CC2CC1C2>>product",  # Non-chiral version, idx 4
            "CC>>CCO",  # Acyclic, no scaffold, idx 5
            "CCC>>CCCO",  # Acyclic, no scaffold, idx 6
            "C1CC1.c1ccccc1>>C1CC1c1ccccc1",  # Multi-component, idx 7
        ],
        "id": range(8),
    }
    return pd.DataFrame(data)


@pytest.fixture
def alkane_dataframe():
    """
    Fixture to create a sample DataFrame for testing SizeSplitter with reaction SMILES.
    It includes a 'smiles' column with reaction SMILES of varying molecule sizes.
    The number of heavy atoms in each reaction is 2*i + 1.
    """
    smiles_data = [f"{'C' * i}>>{'C' * (i + 1)}" for i in range(1, 21)]  # 20 samples
    return pd.DataFrame({"smiles": smiles_data, "id": range(20)})


@pytest.fixture
def varying_size_dataframe():
    """
    Fixture to create a sample DataFrame for testing SizeSplitter with single molecules.
    It includes a 'smiles' column with single molecule SMILES of varying sizes.
    """
    smiles_data = [
        "C",           # 1 heavy atom (methane)
        "CC",          # 2 heavy atoms (ethane)
        "CCC",         # 3 heavy atoms (propane)
        "CCCC",        # 4 heavy atoms (butane)
        "CCCCC",       # 5 heavy atoms (pentane)
        "c1ccccc1",    # 6 heavy atoms (benzene)
        "CCc1ccccc1",  # 8 heavy atoms (ethylbenzene)
        "c1ccc2ccccc2c1",  # 10 heavy atoms (naphthalene)
        "CCN(CC)CC",   # 7 heavy atoms (triethylamine)
        "C1CCCCC1",    # 6 heavy atoms (cyclohexane)
    ]
    return pd.DataFrame({"smiles": smiles_data, "id": range(len(smiles_data))})


@pytest.fixture
def target_splitter_dataframe():
    """Fixture for testing TargetSplitter with a numeric label progression."""
    return pd.DataFrame({"label": range(20), "id": range(20)})
