import pickle

import pandas as pd

try:
    # Python ≥ 3.12
    from typing import override  # type: ignore
except ImportError:
    # Python < 3.12
    from typing_extensions import override  # type: ignore

from chemtorch.components.data_pipeline.data_splitter import DataSplitter
from chemtorch.utils import DataSplit


class IndexSplitter(DataSplitter):
    def __init__(
        self,
        split_index_path: str,
        save_split_dir: str | None = None,
        save_indices: bool = True,
        save_csv: bool = False,
    ):
        """
        Initializes the IndexSplitter with the specified index path.

        Args:
            split_index_path (str): The path to the pickle file containing the index.
            save_split_dir (str | None, optional): If provided, enables saving of split files.
            save_indices (bool): If True and `save_split_dir` is set, re-saves 'indices.pkl'.
            save_csv (bool): If True and `save_split_dir` is set, saves split DataFrames as CSVs.
        """
        super().__init__(
            save_split_dir=save_split_dir, save_indices=save_indices, save_csv=save_csv
        )
        with open(split_index_path, "rb") as f:
            split_indices = pickle.load(f)[0]

        if len(split_indices) != 3:
            raise ValueError(
                "Pickle file must contain exactly 3 arrays for train/val/test splits"
            )

        self.split_map = {
            "train": split_indices[0],
            "val": split_indices[1],
            "test": split_indices[2],
        }

    @override
    def __call__(self, df: pd.DataFrame) -> DataSplit:
        """
        Splits the raw data into training, validation, and test partitions based on the specified indices.

        Args:
            df (pd.DataFrame): The input DataFrame to be split.

        Returns:
            DataSplit: A named tuple containing the train, val, and test dataframes.
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        data_split = DataSplit(
            train=df.iloc[self.split_map["train"]],
            val=df.iloc[self.split_map["val"]],
            test=df.iloc[self.split_map["test"]],
        )

        self._save_split(
            data_split=data_split,
            train_indices=self.split_map["train"],
            val_indices=self.split_map["val"],
            test_indices=self.split_map["test"],
        )

        return data_split
