import pickle

import pandas as pd
from typing_extensions import override

from deepreaction.data_ingestor.data_splitter import DataSplitter
from deepreaction.utils import DataSplit


class IndexSplitter(DataSplitter):
    def __init__(self, split_index_path: str):
        """
        Initializes the IndexSplitter with the specified index path.

        Args:
            split_index_path (str): The path to the pickle file containg the index.
        """
        super(IndexSplitter, self).__init__()
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

        return DataSplit(
            train=df.iloc[self.split_map["train"]],
            val=df.iloc[self.split_map["val"]],
            test=df.iloc[self.split_map["test"]],
        )
