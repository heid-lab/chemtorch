import pandas as pd

try:
    # Python ≥ 3.12
    from typing import override  # type: ignore
except ImportError:
    # Python < 3.12
    from typing_extensions import override  # type: ignore

from chemtorch.components.data_pipeline.data_splitter import DataSplitter
from chemtorch.utils import DataSplit


class TargetSplitter(DataSplitter):
    def __init__(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        sort_order: str = "ascending",
        save_split_dir: str | None = None,
        save_indices: bool = True,
        save_csv: bool = False,
    ):
        """
        Initializes the TargetSplitter.

        Args:
            train_ratio (float): The ratio of data for the training set.
            val_ratio (float): The ratio of data for the validation set.
            test_ratio (float): The ratio of data for the test set.
            sort_order (str): 'ascending' or 'descending'.
            seed (int | None, optional): Random seed for shuffling. Defaults to None.
            save_split_dir (str | None, optional): Directory to save splits.
            save_indices (bool): If True, saves split indices.
            save_csv (bool): If True, saves split DataFrames as CSVs.
        """
        super().__init__(
            save_split_dir=save_split_dir,
            save_indices=save_indices,
            save_csv=save_csv,
        )
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.sort_order = sort_order.lower()

        if not (
            1 - 1e-4 < self.train_ratio + self.val_ratio + self.test_ratio < 1 + 1e-4
        ):
            raise ValueError("Ratios (train, val, test) must sum to approximately 1.")
        if self.sort_order not in ["ascending", "descending"]:
            raise ValueError("sort_order must be 'ascending' or 'descending'.")

    @override
    def __call__(self, df: pd.DataFrame) -> DataSplit:
        """
        Splits the DataFrame based on sorted values of the target_column.

        The DataFrame is first sorted by the target_column. Then it's split into
        train, validation, and test sets according to the specified ratios.
        Finally, each set is shuffled randomly.

        Args:
            df (pd.DataFrame): The input DataFrame to be split.

        Returns:
            DataSplit: A named tuple containing the train, val, and test DataFrames.
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        if "label" not in df.columns:
            raise ValueError(
                f"Target column 'label' not found in DataFrame columns: {df.columns.tolist()}"
            )

        is_ascending = self.sort_order == "ascending"

        sorted_indices = df["label"].sort_values(ascending=is_ascending).index

        n_total = len(df)
        train_size = round(self.train_ratio * n_total)
        val_size = round(self.val_ratio * n_total)

        train_indices = sorted_indices[:train_size]
        val_indices = sorted_indices[train_size : train_size + val_size]
        test_indices = sorted_indices[train_size + val_size :]

        train_df = df.loc[train_indices].sample(frac=1)
        val_df = df.loc[val_indices].sample(frac=1)
        test_df = df.loc[test_indices].sample(frac=1)

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
