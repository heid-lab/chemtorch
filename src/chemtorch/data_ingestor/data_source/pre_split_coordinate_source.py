import os

import pandas as pd
import numpy as np
from typing_extensions import override

from chemtorch.data_ingestor.data_source import DataSource
from chemtorch.utils import DataSplit

def npz_to_df(path: str) -> pd.DataFrame:
    arrs = np.load(path)
    return pd.DataFrame({
        "key":   arrs.files,
        "array": [arrs[k] for k in arrs.files]
    })

class PreSplitCoordinateSource(DataSource):
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

        train_coord = npz_to_df(f"{self.data_folder}/train_mace_mp_ts.npz")
        val_coord   = npz_to_df(f"{self.data_folder}/val_mace_mp_ts.npz")
        test_coord  = npz_to_df(f"{self.data_folder}/test_mace_mp_ts.npz")

        return DataSplit(train=train, val=val, test=test, train_coord=train_coord, val_coord=val_coord, test_coord=test_coord)
