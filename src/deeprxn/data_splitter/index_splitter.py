import pickle
from typing_extensions import override

import pandas as pd
from deeprxn.data import DataSplit, DataSplitter


class IndexSplitter(DataSplitter):
    def __init__(self, index_path: str):
        """
        Initializes the IndexSplitter with the specified index path.

        Args:
            index_path (str): The path to the pickle file containg the index.
        """
        with open(index_path, "rb") as f:
            split_indices = pickle.load(f)
        
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
    def forward(self, raw: pd.DataFrame):
        """
        Splits the raw data into training, validation, and test partitions based on the specified indices.

        Args:
            raw: The raw data to be split.

        Returns:
            DataSplit: A named tuple containing the train, val, and test dataframes.
        """
        return DataSplit(
            train=raw.iloc[self.split_map["train"]],
            val=raw.iloc[self.split_map["val"]],
            test=raw.iloc[self.split_map["test"]],
        )

