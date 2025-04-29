import os
from typing_extensions import override
import pandas as pd
from deeprxn.data import DataReader


class SingleCSVReader(DataReader):
    def __init__(
            self, 
            data_path: str,   
    ):
        self.data_path = data_path

    @override
    def forward(self) -> pd.DataFrame:
        """
        Read data from a single CSV file.
        """
        return pd.read_csv(self.data_path)
