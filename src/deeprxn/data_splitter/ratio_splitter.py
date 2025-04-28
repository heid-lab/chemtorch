from typing import override
import pandas as pd
from deeprxn.data import DataSplit, DataSplitter


class RatioSplitter(DataSplitter):
    
    def __init__(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ):
        """
        Initializes the RatioSplitter with the specified ratios for training, validation, and testing.

        Args:
            train_ratio (float): The ratio of data to be used for training.
            val_ratio (float): The ratio of data to be used for validation.
            test_ratio (float): The ratio of data to be used for testing.
        """
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio


    @override
    def forward(self, raw: pd.DataFrame) -> DataSplit:
        """
        Splits the raw data into training, validation, and test partitions based on the specified ratios.

        Args:
            raw: The raw data to be split.

        Returns:
            DataSplit: A named tuple containing the train, val, and test dataframes.
        """
        assert (
            self.train_ratio + self.val_ratio + self.test_ratio == 1
        ), "Ratios must sum to 1"
        
        # TODO: Ensure random seed it set for reproducibility
        random_df = raw.sample(frac=1).reset_index(drop=True)

        train_size = int(len(random_df) * self.train_ratio)
        val_size = int(len(random_df) * self.val_ratio)

        train_df = random_df[:train_size]
        val_df = random_df[train_size : train_size + val_size]
        test_df = random_df[train_size + val_size :]

        return DataSplit(train=train_df, val=val_df, test=test_df)