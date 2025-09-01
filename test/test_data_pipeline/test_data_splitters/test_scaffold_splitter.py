import pandas as pd
import pytest
import warnings

from chemtorch.components.data_pipeline.data_splitter import ScaffoldSplitter
from chemtorch.utils import DataSplit


def get_scaffolds(df: pd.DataFrame, splitter: ScaffoldSplitter) -> set:
    """Helper to extract non-empty scaffolds from a DataFrame."""
    if df.empty:
        return set()
    scaffolds = df["smiles"].apply(splitter._make_group_id_from_smiles)
    return set(s for s in scaffolds if s)


def test_scaffold_splitter_basic_split(scaffold_dataframe):
    """Test the basic functionality of ScaffoldSplitter with reaction SMILES."""
    splitter = ScaffoldSplitter(
        train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
        split_on="reactant", mol_idx=0  # Required for reaction SMILES
    )
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


def test_scaffold_splitter_single_molecules(single_molecule_dataframe):
    """Test ScaffoldSplitter with single molecules - no split_on or mol_idx needed."""
    splitter = ScaffoldSplitter(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    data_split = splitter(single_molecule_dataframe)

    assert isinstance(data_split, DataSplit)
    total_rows = len(data_split.train) + len(data_split.val) + len(data_split.test)
    assert total_rows == len(single_molecule_dataframe)

    # Check scaffold separation
    train_scaffolds = get_scaffolds(data_split.train, splitter)
    val_scaffolds = get_scaffolds(data_split.val, splitter)
    test_scaffolds = get_scaffolds(data_split.test, splitter)

    assert train_scaffolds.isdisjoint(val_scaffolds)
    assert train_scaffolds.isdisjoint(test_scaffolds)
    assert val_scaffolds.isdisjoint(test_scaffolds)


def test_scaffold_splitter_mixed_molecules_and_reactions():
    """Test ScaffoldSplitter with a mix of single molecules and reactions."""
    mixed_data = pd.DataFrame({
        "smiles": [
            "c1ccccc1",              # Single molecule benzene
            "c1ccccc1>>c1ccncc1",    # Reaction with benzene reactant
            "c1ccncc1",              # Single molecule pyridine
            "CC>>CCO",               # Reaction with acyclic molecules
            "C1CCCCC1.O>>CCO",       # Multi-component reaction
        ],
        "id": range(5)
    })
    
    # For mixed data, we need to provide split_on and mol_idx for the reactions
    splitter = ScaffoldSplitter(
        train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
        split_on="reactant", mol_idx=0
    )
    data_split = splitter(mixed_data)
    
    assert isinstance(data_split, DataSplit)
    total_rows = len(data_split.train) + len(data_split.val) + len(data_split.test)
    assert total_rows == len(mixed_data)


def test_scaffold_splitter_requires_split_on_for_reactions():
    """Test that split_on is required for reaction SMILES."""
    reaction_data = pd.DataFrame({
        "smiles": ["c1ccccc1>>c1ccncc1"],
        "id": [0]
    })
    
    # Should raise error when split_on is not provided for reaction SMILES
    splitter = ScaffoldSplitter(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    with pytest.raises(ValueError, match="split_on parameter is required for reaction SMILES"):
        splitter(reaction_data)


def test_scaffold_splitter_requires_mol_idx_for_multi_component():
    """Test that mol_idx is required for multi-component SMILES."""
    multi_data = pd.DataFrame({
        "smiles": ["C1CC1.c1ccccc1"],  # Multi-component single molecule
        "id": [0]
    })
    
    # Should raise error when mol_idx is not provided for multi-component SMILES
    splitter = ScaffoldSplitter(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    with pytest.raises(ValueError, match="mol_idx parameter is required for multi-component SMILES"):
        splitter(multi_data)


def test_scaffold_splitter_multi_component_single_molecule_with_mol_idx():
    """Test that multi-component single molecules work when mol_idx is provided."""
    multi_data = pd.DataFrame({
        "smiles": ["C1CC1.c1ccccc1"],  # Multi-component single molecule
        "id": [0]
    })
    
    # Should work when mol_idx is provided
    splitter = ScaffoldSplitter(train_ratio=1.0, val_ratio=0.0, test_ratio=0.0, mol_idx=0)
    data_split = splitter(multi_data)
    
    assert isinstance(data_split, DataSplit)
    assert len(data_split.train) == 1


def test_scaffold_splitter_single_component_works_without_mol_idx():
    """Test that single-component molecules work without mol_idx."""
    single_data = pd.DataFrame({
        "smiles": ["c1ccccc1"],  # Single component
        "id": [0]
    })
    
    # Should work fine without mol_idx for single-component molecules
    splitter = ScaffoldSplitter(train_ratio=1.0, val_ratio=0.0, test_ratio=0.0)
    data_split = splitter(single_data)
    
    assert isinstance(data_split, DataSplit)
    assert len(data_split.train) == 1


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
    splitter_chiral = ScaffoldSplitter(include_chirality=True, split_on="reactant", mol_idx=0)
    scaffold_with_stereo = splitter_chiral._make_group_id_from_smiles(chiral_smiles)
    scaffold_from_non_stereo = splitter_chiral._make_group_id_from_smiles(non_chiral_smiles)

    # The scaffold from the chiral SMILES should contain the chiral tags
    assert scaffold_with_stereo == "C1C[C@H]2C[C@@H]1C2"
    # The scaffold from the non-chiral SMILES should not
    assert scaffold_from_non_stereo == "C1CC2CC1C2"
    # Therefore, they must be different
    assert scaffold_with_stereo != scaffold_from_non_stereo

    # CASE 2: include_chirality = False (should strip stereo info)
    splitter_no_chiral = ScaffoldSplitter(include_chirality=False, split_on="reactant", mol_idx=0)
    scaffold_stripped = splitter_no_chiral._make_group_id_from_smiles(chiral_smiles)

    # The scaffold from the chiral SMILES should now be identical to the non-chiral version
    assert scaffold_stripped == "C1CC2CC1C2"
    assert "@" not in scaffold_stripped


def test_scaffold_splitter_split_on_product():
    """Test the `split_on='product'` functionality."""
    splitter = ScaffoldSplitter(split_on="product", mol_idx=0)
    scaffold = splitter._make_group_id_from_smiles("c1ccccc1>>C1CCCCC1")
    assert scaffold == "C1CCCCC1"


def test_scaffold_splitter_mol_idx():
    """Test the `mol_idx` functionality for multi-component SMILES."""
    # Test with reaction SMILES
    smiles = "C1CC1.c1ccccc1>>C1CC1c1ccccc1"

    splitter_first = ScaffoldSplitter(split_on="reactant", mol_idx=0)  # First molecule
    assert splitter_first._make_group_id_from_smiles(smiles) == "C1CC1"

    splitter_second = ScaffoldSplitter(split_on="reactant", mol_idx=1)  # Second molecule
    assert splitter_second._make_group_id_from_smiles(smiles) == "c1ccccc1"

    # Test with single molecule SMILES containing multiple components
    single_multi = "C1CC1.c1ccccc1"
    
    splitter_0 = ScaffoldSplitter(mol_idx=0)
    assert splitter_0._make_group_id_from_smiles(single_multi) == "C1CC1"
    
    splitter_1 = ScaffoldSplitter(mol_idx=1)
    assert splitter_1._make_group_id_from_smiles(single_multi) == "c1ccccc1"


def test_scaffold_splitter_mol_idx_edge_cases():
    """Test mol_idx with edge cases like out of bounds indices."""
    splitter = ScaffoldSplitter(split_on="reactant", mol_idx=5)  # Index out of bounds
    
    with pytest.raises(IndexError, match="out of bounds"):
        splitter._make_group_id_from_smiles("C1CC1.c1ccccc1>>product")


def test_scaffold_splitter_init_errors():
    """Test initialization errors for ScaffoldSplitter."""
    with pytest.raises(ValueError, match="Ratios \\(train, val, test\\) must sum to 1.0"):
        ScaffoldSplitter(train_ratio=0.7, val_ratio=0.2, test_ratio=0.2)

    with pytest.raises(
        ValueError, match="`split_on` must be either 'reactant' or 'product'"
    ):
        ScaffoldSplitter(split_on="catalyst")

    with pytest.raises(
        ValueError, match="`mol_idx` must be a non-negative integer"
    ):
        ScaffoldSplitter(mol_idx=-1)  # Negative index


def test_scaffold_splitter_invalid_inputs():
    """Test splitter behavior with various invalid inputs."""
    splitter = ScaffoldSplitter()

    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        splitter(pd.DataFrame())

    with pytest.raises(ValueError, match="SMILES column 'smiles' not found"):
        splitter(pd.DataFrame({"data": [1, 2, 3]}))

    # Test invalid single molecule SMILES - should raise ValueError
    with pytest.raises(ValueError, match="Could not parse molecule SMILES"):
        splitter._make_group_id_from_smiles("invalid_smiles")
        
    # Test empty string - should raise ValueError
    with pytest.raises(ValueError, match="Invalid SMILES format"):
        splitter._make_group_id_from_smiles("")


def test_scaffold_splitter_invalid_smiles_in_dataframe():
    """Test that invalid SMILES in DataFrame causes the split to fail."""
    invalid_data = pd.DataFrame({
        "smiles": ["c1ccccc1", "invalid_smiles"],  # One valid, one invalid
        "id": range(2)
    })
    
    splitter = ScaffoldSplitter(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    
    # Should raise error when processing the DataFrame with invalid SMILES
    with pytest.raises(ValueError, match="Could not parse molecule SMILES"):
        splitter(invalid_data)


def test_scaffold_splitter_empty_splits_single_molecules():
    """Test ScaffoldSplitter with single molecules and empty validation split."""
    single_mol_df = pd.DataFrame({
        "smiles": ["c1ccccc1", "c1ccncc1", "C1CCCCC1", "c1ccc2ccccc2c1", "C1CC1"],
        "id": range(5)
    })
    
    splitter = ScaffoldSplitter(
        train_ratio=0.8,
        val_ratio=0.0,  # Empty validation
        test_ratio=0.2
    )
    data_split = splitter(single_mol_df)
    
    assert isinstance(data_split, DataSplit)
    assert len(data_split.val) == 0  # Empty validation
    assert data_split.val.empty
    assert data_split.val.shape == (0, 2)  # 0 rows, 2 columns (smiles, id)
    
    total_rows = len(data_split.train) + len(data_split.val) + len(data_split.test)
    assert total_rows == len(single_mol_df)
