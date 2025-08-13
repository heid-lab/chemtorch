import pandas as pd
import pytest
import warnings

from chemtorch.components.data_pipeline.data_splitter import ScaffoldSplitter
from chemtorch.utils import DataSplit


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
            "invalid>>smiles",  # Invalid SMILES, idx 8
        ],
        "id": range(9),
    }
    return pd.DataFrame(data)


def get_scaffolds(df: pd.DataFrame, splitter: ScaffoldSplitter) -> set:
    """Helper to extract non-empty scaffolds from a DataFrame."""
    if df.empty:
        return set()
    scaffolds = df["smiles"].apply(splitter._get_scaffold_smiles)
    return set(s for s in scaffolds if s)


def test_scaffold_splitter_basic_split(scaffold_dataframe):
    """Test the basic functionality of ScaffoldSplitter."""
    splitter = ScaffoldSplitter(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    data_split = splitter(scaffold_dataframe)

    assert isinstance(data_split, DataSplit)
    total_rows = len(data_split.train) + len(data_split.val) + len(data_split.test)
    assert total_rows == len(scaffold_dataframe)

    train_scaffolds = get_scaffolds(data_split.train, splitter)
    val_scaffolds = get_scaffolds(data_split.val, splitter)
    test_scaffolds = get_scaffolds(data_split.test, splitter)

    assert train_scaffolds.isdisjoint(val_scaffolds)
    assert train_scaffolds.isdisjoint(test_scaffolds)
    assert val_scaffolds.isdisjoint(test_scaffolds)

    train_ids = set(data_split.train["id"])
    assert 5 in train_ids  # Acyclic
    assert 6 in train_ids  # Acyclic
    assert 8 in train_ids  # Invalid


def test_include_chirality_flag():
    """
    Test that the `include_chirality` flag correctly differentiates
    or groups stereoisomers, using a molecule where chirality is
    part of the scaffold itself.
    """
    # Norbornane stereoisomer where chirality is integral to the scaffold
    chiral_smiles = "C1C[C@H]2C[C@@H]1C2>>product"
    # The same molecule without stereochemical information
    non_chiral_smiles = "C1CC2CC1C2>>product"

    # CASE 1: include_chirality = True (should preserve stereo info)
    splitter_chiral = ScaffoldSplitter(include_chirality=True)
    scaffold_with_stereo = splitter_chiral._get_scaffold_smiles(chiral_smiles)
    scaffold_from_non_stereo = splitter_chiral._get_scaffold_smiles(non_chiral_smiles)

    # The scaffold from the chiral SMILES should contain the chiral tags
    assert scaffold_with_stereo == "C1C[C@H]2C[C@@H]1C2"
    # The scaffold from the non-chiral SMILES should not
    assert scaffold_from_non_stereo == "C1CC2CC1C2"
    # Therefore, they must be different
    assert scaffold_with_stereo != scaffold_from_non_stereo

    # CASE 2: include_chirality = False (should strip stereo info)
    splitter_no_chiral = ScaffoldSplitter(include_chirality=False)
    scaffold_stripped = splitter_no_chiral._get_scaffold_smiles(chiral_smiles)

    # The scaffold from the chiral SMILES should now be identical to the non-chiral version
    assert scaffold_stripped == "C1CC2CC1C2"
    assert "@" not in scaffold_stripped


def test_scaffold_splitter_split_on_product():
    """Test the `split_on='product'` functionality."""
    splitter = ScaffoldSplitter(split_on="product")
    scaffold = splitter._get_scaffold_smiles("c1ccccc1>>C1CCCCC1")
    assert scaffold == "C1CCCCC1"


def test_scaffold_splitter_mol_idx():
    """Test the `mol_idx` functionality for multi-component SMILES."""
    smiles = "C1CC1.c1ccccc1>>C1CC1c1ccccc1"

    splitter_first = ScaffoldSplitter(mol_idx="first")
    assert splitter_first._get_scaffold_smiles(smiles) == "C1CC1"

    splitter_last = ScaffoldSplitter(mol_idx="last")
    assert splitter_last._get_scaffold_smiles(smiles) == "c1ccccc1"

    splitter_int = ScaffoldSplitter(mol_idx=1)
    assert splitter_int._get_scaffold_smiles(smiles) == "c1ccccc1"


def test_scaffold_splitter_invalid_inputs():
    """Test splitter behavior with various invalid inputs."""
    splitter = ScaffoldSplitter()

    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        splitter(pd.DataFrame())

    with pytest.raises(ValueError, match="SMILES column 'smiles' not found"):
        splitter(pd.DataFrame({"data": [1, 2, 3]}))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        scaffold = splitter._get_scaffold_smiles("this is not a smiles")
        assert scaffold == ""
        assert len(w) >= 1
        assert "Invalid reaction SMILES format" in str(w[-1].message)


def test_scaffold_splitter_init_errors():
    """Test initialization errors for ScaffoldSplitter."""
    with pytest.raises(ValueError, match="Ratios .* must sum to approximately 1"):
        ScaffoldSplitter(train_ratio=0.7, val_ratio=0.2, test_ratio=0.2)

    with pytest.raises(
        ValueError, match="`split_on` must be either 'reactant' or 'product'"
    ):
        ScaffoldSplitter(split_on="catalyst")

    with pytest.raises(
        ValueError, match="`mol_idx` must be an integer, 'first', or 'last'"
    ):
        ScaffoldSplitter(mol_idx="middle")
