from abc import ABC, abstractmethod
import os
import pickle
from typing import Collection
import numpy as np

import pandas as pd

from chemtorch.utils import DataSplit


class DataSplitter(ABC):
    """
    Abstract base class for data splitting strategies.

    Subclass should implement the `__call__` method to define the splitting logic.
    """

    def __init__(
        self,
        save_split_dir: str | None = None,
        save_indices: bool = True,
        save_csv: bool = False,
    ) -> None:
        """
        Initializes the DataSplitter.

        Args:
            save_split_dir (str | None, optional): If provided, enables saving of split files
                to this directory. Defaults to None.
            save_indices (bool): If True and `save_split_dir` is set, saves split indices
                to 'indices.pkl'. Defaults to True.
            save_csv (bool): If True and `save_split_dir` is set, saves split DataFrames
                to 'train.csv', 'val.csv', and 'test.csv'. Defaults to False.
        """
        self.save_split_dir = save_split_dir
        self.save_indices = save_indices
        self.save_csv = save_csv

    @abstractmethod
    def __call__(self, df: pd.DataFrame) -> DataSplit:
        """
        Splits the raw data into training, validation, and test partitions.

        Args:
            df (pd.DataFrame): The raw data to be split.

        Returns:
            DataSplit: A named tuple containing the train, val, and test dataframes.
        """
        pass

    def _save_split(
        self,
        data_split: DataSplit,
        train_indices: Collection,
        val_indices: Collection,
        test_indices: Collection,
    ) -> None:
        """
        Saves split indices and/or DataFrames based on the instance's configuration.

        Saving is only performed if `self.save_split_dir` is not None.
        The directory will be created if it doesn't exist.

        Args:
            data_split (DataSplit): The named tuple containing train, val, test DataFrames.
            train_indices (Collection): The original indices for the training set.
            val_indices (Collection): The original indices for the validation set.
            test_indices (Collection): The original indices for the test set.
        """
        if not self.save_split_dir:
            return

        os.makedirs(self.save_split_dir, exist_ok=True)

        if self.save_indices:
            split_indices = [
                np.array(train_indices),
                np.array(val_indices),
                np.array(test_indices),
            ]
            indices_path = os.path.join(self.save_split_dir, "indices.pkl")
            with open(indices_path, "wb") as f:
                pickle.dump([split_indices], f)

        if self.save_csv:
            data_split.train.to_csv(
                os.path.join(self.save_split_dir, "train.csv"), index=False
            )
            data_split.val.to_csv(
                os.path.join(self.save_split_dir, "val.csv"), index=False
            )
            data_split.test.to_csv(
                os.path.join(self.save_split_dir, "test.csv"), index=False
            )
