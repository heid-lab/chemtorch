import pandas as pd
from torch import nn

from deeprxn.data_pipeline.data_split import DataSplit


class DataSplitter(nn.Module):
    """
    Base class for data splitting strategies.
    """
    def __init__(self):
        """
        Initializes the DataSplitter.
        """
        super(DataSplitter, self).__init__()


    def forward(self, df: pd.DataFrame) -> DataSplit:
        """
        Splits the raw data into training, validation, and test partitions.

        Args:
            df (pd.DataFrame): The raw data to be split.

        Returns:
            DataSplit: A named tuple containing the train, val, and test dataframes.
        """
        raise NotImplementedError("Subclasses should implement this method.")