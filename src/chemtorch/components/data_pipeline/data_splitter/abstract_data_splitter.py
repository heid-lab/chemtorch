from abc import ABC, abstractmethod
import pandas as pd

from chemtorch.utils import DataSplit


class AbstractDataSplitter(ABC):
    """
    Abstract base class for data splitting strategies.

    Subclass should implement the `__call__` method to define the splitting logic.
    """

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
