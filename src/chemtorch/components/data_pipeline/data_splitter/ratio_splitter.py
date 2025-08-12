import pandas as pd

try:
    # Python ≥ 3.12
    from typing import override  # type: ignore
except ImportError:
    # Python < 3.12
    from typing_extensions import override  # type: ignore

from chemtorch.components.data_pipeline.data_splitter import DataSplitter
from chemtorch.utils import DataSplit


class RatioSplitter(DataSplitter):
    def __init__(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        save_split_dir: str | None = None,
        save_indices: bool = True,
        save_csv: bool = False,
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
        super().__init__(
            save_split_dir=save_split_dir, save_indices=save_indices, save_csv=save_csv
        )
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        if not (
            1 - 1e-4 < self.train_ratio + self.val_ratio + self.test_ratio < 1 + 1e-4
        ):
            raise ValueError("Ratios must sum to 1.")

    @override
    def __call__(self, df: pd.DataFrame) -> DataSplit:
        """
        Splits the raw data into training, validation, and test partitions based on the specified ratios.

        Args:
            df (pd.DataFrame): The input DataFrame to be split.

        Returns:
            DataSplit: A named tuple containing the train, val, and test dataframes.
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        random_df = df.sample(frac=1)

        train_size = int(len(random_df) * self.train_ratio)
        val_size = int(len(random_df) * self.val_ratio)

        train_df = random_df[:train_size]
        val_df = random_df[train_size : train_size + val_size]
        test_df = random_df[train_size + val_size :]

        data_split = DataSplit(
            train=train_df.reset_index(drop=True),
            val=val_df.reset_index(drop=True),
            test=test_df.reset_index(drop=True),
        )

        self._save_split(
            data_split=data_split,
            train_indices=train_df.index,
            val_indices=val_df.index,
            test_indices=test_df.index,
        )

        return data_split
