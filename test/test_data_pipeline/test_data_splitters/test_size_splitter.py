import pandas as pd
import pytest

from chemtorch.components.data_pipeline.data_splitter import SizeSplitter
from chemtorch.utils import DataSplit


@pytest.fixture
def size_splitter_dataframe():
    """
    Fixture to create a sample DataFrame for testing SizeSplitter.
    It includes a 'smiles' column with reaction SMILES of varying molecule sizes.
    The number of heavy atoms in each reaction is 2*i + 1.
    """
    smiles_data = [f"{'C' * i}>>{'C' * (i + 1)}" for i in range(1, 21)]  # 20 samples
    return pd.DataFrame({"smiles": smiles_data, "id": range(20)})


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


def test_get_n_heavy_atoms_helper():
    """Test the internal _get_n_heavy_atoms helper function directly."""
    splitter = SizeSplitter()
    assert splitter._get_n_heavy_atoms("C>>CC") == 3
    assert splitter._get_n_heavy_atoms("O>>[O-]") == 2
    assert splitter._get_n_heavy_atoms("c1ccccc1>>c1ccncc1") == 12

    with pytest.raises(ValueError, match="Invalid SMILES string"):
        splitter._get_n_heavy_atoms(None)
    with pytest.raises(Exception):
        splitter._get_n_heavy_atoms("not_a_smiles>>still_not_a_smiles")


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
    with pytest.raises(ValueError, match="Ratios .* must sum to approximately 1"):
        SizeSplitter(train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)  # Sum > 1
    with pytest.raises(ValueError, match="Ratios .* must sum to approximately 1"):
        SizeSplitter(train_ratio=0.5, val_ratio=0.2, test_ratio=0.2)  # Sum < 1


def test_size_splitter_invalid_sort_order():
    """Test SizeSplitter initialization with an invalid sort_order."""
    with pytest.raises(
        ValueError, match="sort_order must be 'ascending' or 'descending'"
    ):
        SizeSplitter(sort_order="random")
