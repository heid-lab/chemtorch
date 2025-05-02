import os
import pandas as pd
from typing_extensions import override
from deeprxn.data_pipeline.data_reader.data_reader import DataReader
from deeprxn.data_pipeline.data_pipeline import DataSplit


class SplitCSVReader(DataReader):
    def __init__(
            self,
            data_folder: str,
    ):
        self.data_folder = data_folder

    @override
    def forward(self) -> DataSplit:
        """
        Read data from a CSV file.
        """
        files = {
            name: os.path.join(self.data_folder, f"{name}.csv")
            for name in ["train", "val", "test"]
        }
        missing_files = [
            file for file in files.values() if not os.path.exists(file)
        ]
        if missing_files:
            raise FileNotFoundError(f"Missing files:\n{chr(10).join(missing_files)}")

        train = pd.read_csv(files["train"])
        val = pd.read_csv(files["val"])
        test = pd.read_csv(files["test"])
        return DataSplit(train=train, val=val, test=test)
