from abc import ABC, abstractmethod
import os
from pathlib import Path
import pickle
from typing import Collection, List
import numpy as np

import pandas as pd

from chemtorch.utils import DataSplit
from chemtorch.components.data_pipeline.data_splitter.abstract_data_splitter import AbstractDataSplitter


class DataSplitterBase(AbstractDataSplitter):
    """
    Base class for data splitting strategies.
    Callable that takes a DataFrame, executes splitting logic, and returns a DataSplit object.
    Subclass should implement the private `_split` method to define specific splitting logic.
    """

    def __init__(
        self,
        save_path: str | None = None,
    ) -> None:
        """
        Initializes the DataSplitter.

        Args:
            save_path (str | None, optional): If provided, saves split indices as a pickle file
                to this path. Must end with '.pkl'. Defaults to None.

        Raises:
            ValueError: If the provided `save_path` is not a pickle file (i.e. does not end with '.pkl').
        """
        if save_path is not None and not save_path.endswith(".pkl"):
            raise ValueError("save_path must end with '.pkl' if provided.")

        self.save_path = Path(save_path) if save_path else None

    def __call__(self, df: pd.DataFrame) -> DataSplit[pd.DataFrame]:
        """
        Splits the raw data into training, validation, and test partitions.

        Args:
            df (pd.DataFrame): The raw data to be split.

        Returns:
            DataSplit: A named tuple containing the train, val, and test dataframes.
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        # NOTE: The random state has to be restored so subsequent operations like data shuffling
        # in the data loader get the same RNG sequence as if no splitting had occurred. This is
        # needed for reproducibility when using IndexSplitter on the saved indices.
        np_state = np.random.get_state()
        indices = self._split(df)
        np.random.set_state(np_state)

        self._save_split(indices)
        return DataSplit(
            train=df.iloc[indices.train],
            val=df.iloc[indices.val],
            test=df.iloc[indices.test],
        )

    @abstractmethod
    def _split(self, df: pd.DataFrame) -> DataSplit[List[int]]:
        """
        Splits the DataFrame into train, validation, and test sets.

        Args:
            df (pd.DataFrame): The input DataFrame to be split.

        Returns:
            DataSplit[Collection]: A named tuple containing the train, val, and test indices.
        """
        pass

    def _save_split(self, indices: DataSplit[List[int]]) -> None:
        """
        Saves split indices and/or DataFrames based on the instance's configuration.

        Saving is only performed if `self.save_split_dir` is not None.
        The directory will be created if it doesn't exist.

        Args:
            train_indices (Collection): The original indices for the training set.
            val_indices (Collection): The original indices for the validation set.
            test_indices (Collection): The original indices for the test set.
        """
        if not self.save_path:
            return

        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.save_path, "wb") as f:
            pickle.dump([indices.to_dict()], f)
