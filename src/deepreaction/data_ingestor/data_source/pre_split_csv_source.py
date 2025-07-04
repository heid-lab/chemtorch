import os

import pandas as pd
from typing_extensions import override

from deepreaction.data_ingestor.data_source import DataSource
from deepreaction.utils import DataSplit


class PreSplitCSVSource(DataSource):
    def __init__(
        self,
        data_folder: str,
    ):
        self.data_folder = data_folder

    @override
    def load(self) -> DataSplit:
        """
        Load presplit data from CSV files in a specified folder.
        The files should be named 'train.csv', 'val.csv', and 'test.csv'.
        """
        files = {
            name: os.path.join(self.data_folder, f"{name}.csv")
            for name in ["train", "val", "test"]
        }
        missing_files = [
            file for file in files.values() if not os.path.exists(file)
        ]
        if missing_files:
            raise FileNotFoundError(
                f"Missing files:\n{chr(10).join(missing_files)}"
            )

        train = pd.read_csv(files["train"])
        val = pd.read_csv(files["val"])
        test = pd.read_csv(files["test"])
        return DataSplit(train=train, val=val, test=test)
