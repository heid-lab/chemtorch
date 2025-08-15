import math
from typing import List
import pandas as pd

try:
    # Python ≥ 3.12
    from typing import override  # type: ignore
except ImportError:
    # Python < 3.12
    from typing_extensions import override  # type: ignore

from chemtorch.components.data_pipeline.data_splitter import DataSplitterBase
from chemtorch.utils import DataSplit


class TargetSplitter(DataSplitterBase):
    def __init__(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        sort_order: str = "ascending",
        save_path: str | None = None,
    ):
        """
        Initializes the TargetSplitter.

        Args:
            train_ratio (float): The ratio of data for the training set.
            val_ratio (float): The ratio of data for the validation set.
            test_ratio (float): The ratio of data for the test set.
            sort_order (str): 'ascending' or 'descending'.
            save_path (str | None, optional): Path to save split indices.
        """
        super().__init__(save_path=save_path)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.sort_order = sort_order.lower()

        ratio_sum = self.train_ratio + self.val_ratio + self.test_ratio
        if not math.isclose(ratio_sum, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"Ratios (train, val, test) must sum to 1.0, got {ratio_sum}")
        
        if self.sort_order not in ["ascending", "descending"]:
            raise ValueError("sort_order must be 'ascending' or 'descending'.")

    @override
    def _split(self, df: pd.DataFrame) -> DataSplit[List[int]]:
        """
        Splits the DataFrame based on sorted values of the label column.

        Args:
            df (pd.DataFrame): The input DataFrame to be split.

        Returns:
            DataSplit[List[int]]: A named tuple containing the train, val, and test indices.
        """
        if "label" not in df.columns:
            raise ValueError(
                f"Target column 'label' not found in DataFrame columns: {df.columns.tolist()}"
            )

        is_ascending = self.sort_order == "ascending"
        sorted_indices = df["label"].sort_values(ascending=is_ascending).index

        n_total = len(df)
        train_size = int(n_total * self.train_ratio)
        val_size = int(n_total * self.val_ratio)

        train_indices = sorted_indices[:train_size].tolist()
        val_indices = sorted_indices[train_size : train_size + val_size].tolist()
        test_indices = sorted_indices[train_size + val_size :].tolist()

        return DataSplit(
            train=train_indices,
            val=val_indices,
            test=test_indices,
        )
