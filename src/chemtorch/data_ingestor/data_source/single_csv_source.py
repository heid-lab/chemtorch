import os

import pandas as pd
from typing_extensions import override

from chemtorch.data_ingestor.data_source.data_source import DataSource


class SingleCSVSource(DataSource):
    def __init__(
        self,
        data_path: str,
    ):
        self.data_path = data_path

    @override
    def load(self) -> pd.DataFrame:
        """
        Load data from a single CSV file.
        """
        return pd.read_csv(self.data_path)
