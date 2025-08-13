import pandas as pd
import pytest

from chemtorch.components.data_pipeline.data_splitter import TargetSplitter
from chemtorch.utils import DataSplit


@pytest.fixture
def target_splitter_dataframe():
    """
    Fixture to create a sample DataFrame for testing TargetSplitter.
    It includes a 'label' column with a clear numeric progression.
    """
    return pd.DataFrame({"label": range(20), "id": range(20)})


def test_target_splitter_ascending(target_splitter_dataframe):
    """
    Test TargetSplitter with ascending sort order.
    The rows with the smallest label values should be in the train set,
    and the largest in the test set.
    """
    splitter = TargetSplitter(
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        sort_order="ascending",
    )
    data_split = splitter(target_splitter_dataframe)

    assert isinstance(data_split, DataSplit)
    total_rows = len(data_split.train) + len(data_split.val) + len(data_split.test)
    assert total_rows == len(target_splitter_dataframe)
    assert len(data_split.train) == 16  # 0.8 * 20
    assert len(data_split.val) == 2  # 0.1 * 20
    assert len(data_split.test) == 2  # 0.1 * 20

    assert data_split.train["id"].max() < data_split.val["id"].min()
    assert data_split.val["id"].max() < data_split.test["id"].min()


def test_target_splitter_descending(target_splitter_dataframe):
    """
    Test TargetSplitter with descending sort order.
    The rows with the largest label values should be in the train set.
    """
    splitter = TargetSplitter(
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        sort_order="descending",
    )
    data_split = splitter(target_splitter_dataframe)

    assert isinstance(data_split, DataSplit)
    total_rows = len(data_split.train) + len(data_split.val) + len(data_split.test)
    assert total_rows == len(target_splitter_dataframe)

    assert data_split.train["id"].min() > data_split.val["id"].max()
    assert data_split.val["id"].min() > data_split.test["id"].max()


def test_target_splitter_empty_dataframe():
    """Test TargetSplitter with an empty DataFrame."""
    splitter = TargetSplitter()
    empty_df = pd.DataFrame({"label": [], "id": []})
    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        splitter(empty_df)


def test_target_splitter_missing_column():
    """Test TargetSplitter with a DataFrame that lacks the label column."""
    splitter = TargetSplitter()
    df = pd.DataFrame({"data": [1, 2, 3]})
    with pytest.raises(ValueError, match="Target column 'label' not found"):
        splitter(df)


def test_target_splitter_invalid_ratios():
    """Test TargetSplitter initialization with invalid ratios."""
    with pytest.raises(ValueError, match="Ratios .* must sum to approximately 1"):
        TargetSplitter(train_ratio=0.6, val_ratio=0.3, test_ratio=0.2)
    with pytest.raises(ValueError, match="Ratios .* must sum to approximately 1"):
        TargetSplitter(train_ratio=0.5, val_ratio=0.2, test_ratio=0.2)


def test_target_splitter_invalid_sort_order():
    """Test TargetSplitter initialization with an invalid sort_order."""
    with pytest.raises(
        ValueError, match="sort_order must be 'ascending' or 'descending'"
    ):
        TargetSplitter(sort_order="alphabetical")
