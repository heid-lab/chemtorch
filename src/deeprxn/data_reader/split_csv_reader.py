from typing import Optional, Tuple, override

import pandas as pd
from deeprxn.data import DataReader, DataSplit


class SplitCSVReader(DataReader):
    def __init__(
            self,
            data_path: str,
    ):
        self.data_path = data_path

    @override
    def forward(self) -> DataSplit:
        """
        Read data from a CSV file.
        """
        train_df = pd.read_csv(f"{self.data_path}/train.csv")
        val_df = pd.read_csv(f"{self.data_path}/val.csv")
        test_df = pd.read_csv(f"{self.data_path}/test.csv")

        return DataSplit(
            train=train_df,
            val=val_df,
            test=test_df,
        )
