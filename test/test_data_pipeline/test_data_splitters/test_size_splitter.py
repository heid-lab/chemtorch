import pandas as pd
import pytest

from chemtorch.components.data_pipeline.data_splitter import SizeSplitter
from chemtorch.utils import DataSplit


@pytest.fixture
def size_splitter_dataframe():
    """
    Fixture to create a sample DataFrame for testing SizeSplitter with reaction SMILES.
    It includes a 'smiles' column with reaction SMILES of varying molecule sizes.
    The number of heavy atoms in each reaction is 2*i + 1.
    """
    smiles_data = [f"{'C' * i}>>{'C' * (i + 1)}" for i in range(1, 21)]  # 20 samples
    return pd.DataFrame({"smiles": smiles_data, "id": range(20)})


@pytest.fixture
def single_molecule_dataframe():
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


def test_size_splitter_ascending(size_splitter_dataframe):
    """
    Test SizeSplitter with ascending sort order.
    The smallest molecules should be in the train set, and the largest in the test set.
    """
    splitter = SizeSplitter(
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        sort_order="ascending",
    )
    data_split = splitter(size_splitter_dataframe)

    assert isinstance(data_split, DataSplit)
    total_rows = len(data_split.train) + len(data_split.val) + len(data_split.test)
    assert total_rows == len(size_splitter_dataframe)
    assert len(data_split.train) == 16  # 0.8 * 20
    assert len(data_split.val) == 2  # 0.1 * 20
    assert len(data_split.test) == 2  # 0.1 * 20
    assert "_mol_size" not in data_split.train.columns

    get_size = splitter._get_n_heavy_atoms
    train_sizes = data_split.train["smiles"].apply(get_size)
    val_sizes = data_split.val["smiles"].apply(get_size)
    test_sizes = data_split.test["smiles"].apply(get_size)

    assert train_sizes.max() <= val_sizes.min()
    assert val_sizes.max() <= test_sizes.min()


def test_size_splitter_descending(size_splitter_dataframe):
    """
    Test SizeSplitter with descending sort order.
    The largest molecules should be in the train set, and the smallest in the test set.
    """
    splitter = SizeSplitter(
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        sort_order="descending",
    )
    data_split = splitter(size_splitter_dataframe)

    assert isinstance(data_split, DataSplit)
    total_rows = len(data_split.train) + len(data_split.val) + len(data_split.test)
    assert total_rows == len(size_splitter_dataframe)
    assert "_mol_size" not in data_split.train.columns

    get_size = splitter._get_n_heavy_atoms
    train_sizes = data_split.train["smiles"].apply(get_size)
    val_sizes = data_split.val["smiles"].apply(get_size)
    test_sizes = data_split.test["smiles"].apply(get_size)

    assert train_sizes.min() >= val_sizes.max()
    assert val_sizes.min() >= test_sizes.max()


def test_size_splitter_single_molecules_ascending(single_molecule_dataframe):
    """
    Test SizeSplitter with single molecules in ascending order.
    Smaller molecules should be in train, larger in test.
    """
    splitter = SizeSplitter(
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        sort_order="ascending",
    )
    data_split = splitter(single_molecule_dataframe)

    assert isinstance(data_split, DataSplit)
    total_rows = len(data_split.train) + len(data_split.val) + len(data_split.test)
    assert total_rows == len(single_molecule_dataframe)
    assert "_mol_size" not in data_split.train.columns

    get_size = splitter._get_n_heavy_atoms
    train_sizes = data_split.train["smiles"].apply(get_size)
    val_sizes = data_split.val["smiles"].apply(get_size)
    test_sizes = data_split.test["smiles"].apply(get_size)

    # Check that train has smaller molecules than val, and val has smaller than test
    assert train_sizes.max() <= val_sizes.min()
    assert val_sizes.max() <= test_sizes.min()


def test_size_splitter_single_molecules_descending(single_molecule_dataframe):
    """
    Test SizeSplitter with single molecules in descending order.
    Larger molecules should be in train, smaller in test.
    """
    splitter = SizeSplitter(
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        sort_order="descending",
    )
    data_split = splitter(single_molecule_dataframe)

    assert isinstance(data_split, DataSplit)
    total_rows = len(data_split.train) + len(data_split.val) + len(data_split.test)
    assert total_rows == len(single_molecule_dataframe)
    assert "_mol_size" not in data_split.train.columns

    get_size = splitter._get_n_heavy_atoms
    train_sizes = data_split.train["smiles"].apply(get_size)
    val_sizes = data_split.val["smiles"].apply(get_size)
    test_sizes = data_split.test["smiles"].apply(get_size)

    # Check that train has larger molecules than val, and val has larger than test
    assert train_sizes.min() >= val_sizes.max()
    assert val_sizes.min() >= test_sizes.max()


def test_size_splitter_mixed_molecules_and_reactions():
    """
    Test SizeSplitter with a mix of single molecules and reactions.
    """
    mixed_data = pd.DataFrame({
        "smiles": [
            "C",              # 1 heavy atom
            "CC>>CCC",        # 2 + 3 = 5 heavy atoms  
            "c1ccccc1",       # 6 heavy atoms
            "O>>CCO",         # 1 + 3 = 4 heavy atoms
            "CCCC",           # 4 heavy atoms
            "N>>CCN",         # 1 + 3 = 4 heavy atoms
        ],
        "id": range(6)
    })
    
    splitter = SizeSplitter(
        train_ratio=0.5,
        val_ratio=0.3,
        test_ratio=0.2,
        sort_order="ascending",
    )
    data_split = splitter(mixed_data)

    assert isinstance(data_split, DataSplit)
    total_rows = len(data_split.train) + len(data_split.val) + len(data_split.test)
    assert total_rows == len(mixed_data)
    
    # Verify that no temporary columns remain
    assert "_mol_size" not in data_split.train.columns
    assert "_mol_size" not in data_split.val.columns
    assert "_mol_size" not in data_split.test.columns


def test_get_n_heavy_atoms_helper():
    """Test the internal _get_n_heavy_atoms helper function directly."""
    splitter = SizeSplitter()
    
    # Test reaction SMILES
    assert splitter._get_n_heavy_atoms("C>>CC") == 3
    assert splitter._get_n_heavy_atoms("O>>[O-]") == 2
    assert splitter._get_n_heavy_atoms("c1ccccc1>>c1ccncc1") == 12
    
    # Test single molecule SMILES
    assert splitter._get_n_heavy_atoms("C") == 1      # methane
    assert splitter._get_n_heavy_atoms("CC") == 2     # ethane
    assert splitter._get_n_heavy_atoms("c1ccccc1") == 6  # benzene
    assert splitter._get_n_heavy_atoms("CCO") == 3    # ethanol
    assert splitter._get_n_heavy_atoms("CCN") == 3    # ethylamine
    assert splitter._get_n_heavy_atoms("c1ccc2ccccc2c1") == 10  # naphthalene

    # Test error cases
    with pytest.raises(ValueError, match="Invalid SMILES string"):
        splitter._get_n_heavy_atoms(None)
    with pytest.raises(ValueError, match="Invalid SMILES string"):
        splitter._get_n_heavy_atoms("")
    with pytest.raises(ValueError, match="Invalid SMILES string"):
        splitter._get_n_heavy_atoms("   ")  # whitespace only
    with pytest.raises(Exception):
        splitter._get_n_heavy_atoms("not_a_smiles>>still_not_a_smiles")
    with pytest.raises(Exception):
        splitter._get_n_heavy_atoms("invalid_single_molecule_smiles")
    
    # Test invalid SMILES
    with pytest.raises(Exception):
        splitter._get_n_heavy_atoms("not_a_smiles>>still_not_a_smiles")
    with pytest.raises(Exception):
        splitter._get_n_heavy_atoms("invalid_single_molecule_smiles")


def test_size_splitter_empty_dataframe():
    """Test SizeSplitter with an empty DataFrame."""
    splitter = SizeSplitter()
    empty_df = pd.DataFrame({"smiles": []})
    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        splitter(empty_df)


def test_size_splitter_missing_smiles_column():
    """Test SizeSplitter with a DataFrame that lacks the 'smiles' column."""
    splitter = SizeSplitter()
    df_no_smiles = pd.DataFrame({"data": [1, 2, 3]})
    with pytest.raises(
        ValueError, match="SMILES column 'smiles' not found in DataFrame"
    ):
        splitter(df_no_smiles)


def test_size_splitter_invalid_ratios():
    """Test SizeSplitter initialization with invalid ratios."""
    with pytest.raises(ValueError, match="Ratios \\(train, val, test\\) must sum to 1.0"):
        SizeSplitter(train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)  # Sum > 1
    with pytest.raises(ValueError, match="Ratios \\(train, val, test\\) must sum to 1.0"):
        SizeSplitter(train_ratio=0.5, val_ratio=0.2, test_ratio=0.2)  # Sum < 1


def test_size_splitter_invalid_sort_order():
    """Test SizeSplitter initialization with an invalid sort_order."""
    with pytest.raises(
        ValueError, match="sort_order must be 'ascending' or 'descending'"
    ):
        SizeSplitter(sort_order="random")


def test_size_splitter_edge_cases_single_molecules():
    """Test SizeSplitter with edge cases for single molecules."""
    # Test with very small molecules
    small_molecules_df = pd.DataFrame({
        "smiles": ["C", "N", "O", "[H]", "CC"],  # Very simple molecules
        "id": range(5)
    })
    
    splitter = SizeSplitter(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    data_split = splitter(small_molecules_df)
    
    assert isinstance(data_split, DataSplit)
    total_rows = len(data_split.train) + len(data_split.val) + len(data_split.test)
    assert total_rows == len(small_molecules_df)


def test_size_splitter_empty_splits_single_molecules():
    """Test SizeSplitter with single molecules and empty validation split."""
    single_mol_df = pd.DataFrame({
        "smiles": ["C", "CC", "CCC", "CCCC", "CCCCC"],
        "id": range(5)
    })
    
    splitter = SizeSplitter(
        train_ratio=0.8,
        val_ratio=0.0,  # Empty validation
        test_ratio=0.2,
        sort_order="ascending"
    )
    data_split = splitter(single_mol_df)
    
    assert isinstance(data_split, DataSplit)
    assert len(data_split.train) == 4  # 0.8 * 5
    assert len(data_split.val) == 0    # Empty
    assert len(data_split.test) == 1   # 0.2 * 5
    
    # Verify val is empty DataFrame, not None
    assert data_split.val.empty
    assert data_split.val.shape == (0, 2)  # 0 rows, 2 columns (smiles, id)
