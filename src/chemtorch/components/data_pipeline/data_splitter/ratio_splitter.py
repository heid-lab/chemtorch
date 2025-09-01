import math
import numpy as np
from typing import Collection, List
import pandas as pd

try:
    # Python ≥ 3.12
    from typing import override  # type: ignore
except ImportError:
    # Python < 3.12
    from typing_extensions import override  # type: ignore

from chemtorch.components.data_pipeline.data_splitter import DataSplitterBase
from chemtorch.utils import DataSplit


class RatioSplitter(DataSplitterBase):
    """
    Splits data into training, validation, and test sets based on specified ratios.

    Subclasses should override the `_split` method to implement custom splitting
    logic. By default the `RatioSplitter` randomly shuffles the data and splits it
    according to the specified ratios.
    """
    def __init__(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        save_path: str | None = None,
    ):
        """
        Initializes a RatioSplitter.

        Args:
            train_ratio (float): The ratio of data for training.
            val_ratio (float): The ratio of data for validation.
            test_ratio (float): The ratio of data for testing.
            save_path (str | None, optional): If provided, saves split indices as pickle file.
        """
        super().__init__(save_path=save_path)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        ratio_sum = self.train_ratio + self.val_ratio + self.test_ratio
        if not math.isclose(ratio_sum, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"Ratios (train, val, test) must sum to 1.0, got {ratio_sum}")

    @override
    def _split(self, df: pd.DataFrame) -> DataSplit[List[int]]:

        n_total = len(df)
        train_size = int(n_total * self.train_ratio)
        val_size = int(n_total * self.val_ratio)

        index = df.sample(frac=1).index.tolist()
        train_idx = index[:train_size]
        val_idx = index[train_size : train_size + val_size]
        test_idx = index[train_size + val_size :]

        return DataSplit(
            train=train_idx,
            val=val_idx,
            test=test_idx,
        )