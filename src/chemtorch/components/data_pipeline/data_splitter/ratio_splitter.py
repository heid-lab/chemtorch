import math
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
    def __init__(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        save_path: str | None = None,
    ):
        """
        Initializes the RatioSplitter.

        Args:
            train_ratio (float): The ratio of data for training.
            val_ratio (float): The ratio of data for validation.
            test_ratio (float): The ratio of data for testing.
            save_split_dir (str | None, optional): If provided, enables saving of split files.
            save_indices (bool): If True and `save_split_dir` is set, saves 'indices.pkl'.
            save_csv (bool): If True and `save_split_dir` is set, saves split DataFrames as CSVs.
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
        random_df = df.sample(frac=1)

        train_size = int(len(random_df) * self.train_ratio)
        val_size = int(len(random_df) * self.val_ratio)

        train_df = random_df[:train_size]
        val_df = random_df[train_size : train_size + val_size]
        test_df = random_df[train_size + val_size :]
        return DataSplit(
            train=train_df.index.tolist(),
            val=val_df.index.tolist(),
            test=test_df.index.tolist(),
        )

        
