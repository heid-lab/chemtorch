import pickle
import warnings

import pandas as pd
from typing import List, Dict
import warnings

try:
    # Python ≥ 3.12
    from typing import override  # type: ignore
except ImportError:
    # Python < 3.12
    from typing_extensions import override  # type: ignore

from chemtorch.components.data_pipeline.data_splitter.abstract_data_splitter import AbstractDataSplitter
from chemtorch.utils import DataSplit


class IndexSplitter(AbstractDataSplitter):
    def __init__(self, split_index_path: str):
        self,
        split_index_path: str,
        save_path: str | None = None,
    ):
        """
        Initializes the IndexSplitter with the specified index path.

        Args:
            split_index_path (str): The path to the pickle file containing the index.
            save_split_dir (str | None, optional): If provided, enables saving of split files.
            save_indices (bool): If True and `save_split_dir` is set, re-saves 'indices.pkl'.
            save_csv (bool): If True and `save_split_dir` is set, saves split DataFrames as CSVs.
        """
        super().__init__(save_path=save_path)
        self.split_map = self._init_split_map(split_index_path)

    def _init_split_map(self, split_index_path: str) -> Dict[str, List[int]]:
        with open(split_index_path, "rb") as f:
            split_indices = pickle.load(f)[0]

        if len(split_indices) != 3:
            raise ValueError(
                "Pickle file must contain exactly 3 arrays for train/val/test splits"
            )

        if isinstance(split_indices, list):
            split_map = {
                "train": split_indices[0],
                "val": split_indices[1],
                "test": split_indices[2],
            }
        elif isinstance(split_indices, dict):
            # check that the dict has keys train, val, test
            if not all(key in split_indices for key in ["train", "val", "test"]):
                raise ValueError(
                    "Dict must contain keys 'train', 'val', and 'test', but got keys: "
                    f"{list(split_indices.keys())}"
                )
            split_map = {
                "train": split_indices["train"],
                "val": split_indices["val"],
                "test": split_indices["test"],
            }
        else:
            raise ValueError("Invalid format for split indices")

        if not all(
            isinstance(indices, list) for indices in split_map.values()
        ):
            raise ValueError("All split indices must be lists")

        if not all(
            all(isinstance(idx, int) for idx in indices) for indices in split_map.values()
        ):
            raise ValueError("All split index lists must contain integers only")

        return split_map

    @override
    def _split(self, df: pd.DataFrame) -> DataSplit[List[int]]:
        """
        Splits the DataFrame into train, validation, and test sets based on pre-defined indices.

        Args:
            df (pd.DataFrame): The input DataFrame to be split.

        Returns:
            DataSplit[List[int]]: A named tuple containing the train, val, and test indices.
        """
        # Check for out-of-bounds indices
        all_indices = []
        for split_name, indices in self.split_map.items():
            all_indices.extend(indices)
            # Check if any index is out of bounds
            max_idx = max(indices) if indices else -1
            if max_idx >= len(df):
                raise ValueError(
                    f"Index {max_idx} in {split_name} split is out of bounds for DataFrame with {len(df)} rows"
                )
        
        # Check for duplicate indices across splits
        if len(all_indices) != len(set(all_indices)):
            raise ValueError("Duplicate indices found across different splits")
        
        # Warn if total number of indices doesn't match DataFrame length
        if len(df) != len(all_indices):
            train_ratio = len(self.split_map["train"]) / len(df)
            val_ratio = len(self.split_map["val"]) / len(df)
            test_ratio = len(self.split_map["test"]) / len(df)
            warnings.warn(
                f"Dataset is implicitly subsampled by the given split index: "
                f"train={train_ratio:.4f}, val={val_ratio:.4f}, test={test_ratio:.4f}"
            )

        return DataSplit(
            train=self.split_map["train"],
            val=self.split_map["val"],
            test=self.split_map["test"],
        )
