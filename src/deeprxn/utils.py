import os
import pickle
import random
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = (
            ":4096:8"  # https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
        )


def load_csv_dataset(
    input_column: str,
    target_column: str,
    data_folder: str,
    reduced_dataset: Union[int, float],
    split: Literal["train", "val", "test"] = "train",
    data_root: str = "data",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    use_pickle: bool = False,
    use_enthalpy: bool = False,
    enthalpy_column: Optional[str] = None,
):
    """ """
    base_path = Path(data_root) / data_folder
    split_files = {s: base_path / f"{s}.csv" for s in ["train", "val", "test"]}
    single_file = base_path / "data.csv"
    pickle_file = base_path / "seed0.pkl"

    if all(f.exists() for f in split_files.values()):
        data_path = split_files[split]
        data_df = pd.read_csv(data_path)
    elif single_file.exists():
        if not np.isclose(sum([train_ratio, val_ratio, test_ratio]), 1.0):
            raise ValueError("Train, val, and test ratios must sum to 1.0")
        data_df = pd.read_csv(single_file)

        if use_pickle:
            if not pickle_file.exists():
                raise FileNotFoundError(
                    f"Pickle file not found at {pickle_file}"
                )

            with open(pickle_file, "rb") as f:
                split_indices = pickle.load(f)[0]

            if len(split_indices) != 3:
                raise ValueError(
                    "Pickle file must contain exactly 3 arrays for train/val/test splits"
                )

            split_map = {
                "train": split_indices[0],
                "val": split_indices[1],
                "test": split_indices[2],
            }
            data_df = data_df.iloc[split_map[split]]

        else:
            data_df = data_df.sample(frac=1)

            n = len(data_df)
            train_idx = int(n * train_ratio)
            val_idx = train_idx + int(n * val_ratio)

            if split == "train":
                data_df = data_df.iloc[:train_idx]
            elif split == "val":
                data_df = data_df.iloc[train_idx:val_idx]
            elif split == "test":
                data_df = data_df.iloc[val_idx:]
    else:
        raise FileNotFoundError(
            f"No dataset found in {base_path}. "
            f"Expected either split files or a single file."
        )

    if reduced_dataset < 1 and split == "train":
        data_df = data_df.sample(int(len(data_df) * reduced_dataset))
    elif reduced_dataset > 1 and split == "train":
        data_df = data_df.sample(int(reduced_dataset))

    missing_cols = []
    required_cols = [input_column, target_column]
    if use_enthalpy:
        if not enthalpy_column:
            raise ValueError(
                "enthalpy_column must be specified when use_enthalpy is True"
            )
        required_cols.append(enthalpy_column)

    for col in required_cols:
        if col not in data_df.columns:
            missing_cols.append(col)

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    inputs = data_df[input_column].values
    targets = data_df[target_column].values.astype(float)

    if use_enthalpy:
        enthalpy = data_df[enthalpy_column].values.astype(float)
        return inputs, targets, enthalpy

    return inputs, targets


def save_model(model, optimizer, epoch, best_val_loss, model_dir):
    """Save model and optimizer state to the model directory."""
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pt")

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        },
        model_path,
    )


def load_model(model, optimizer, model_dir):
    """Load model and optimizer state from the model directory."""

    if os.path.exists(model_dir):
        model_path = os.path.join(model_dir, "model.pt")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint["epoch"]
        best_val_loss = checkpoint["best_val_loss"]

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return model, optimizer, epoch, best_val_loss
    else:
        return model, optimizer, 0, float("inf")


class Standardizer:
    """Standardize data by computing (x - mean) / std."""

    def __init__(
        self, mean: Union[float, np.ndarray], std: Union[float, np.ndarray]
    ):
        """Initialize standardizer with mean and standard deviation.

        Args:
            mean: Mean value(s) for standardization
            std: Standard deviation value(s) for standardization
        """
        self.mean = mean
        self.std = std

    def __call__(
        self, x: Union[torch.Tensor, np.ndarray], rev: bool = False
    ) -> Union[torch.Tensor, np.ndarray]:
        """Apply standardization or reverse standardization.

        Args:
            x: Input data to standardize
            rev: If True, reverse the standardization

        Returns:
            Standardized or reverse standardized data
        """
        if rev:
            return (x * self.std) + self.mean
        return (x - self.mean) / self.std


def save_standardizer(mean, std, model_dir):
    """Save standardizer parameters to the model directory."""
    os.makedirs(model_dir, exist_ok=True)
    standardizer_path = os.path.join(model_dir, "standardizer.pt")
    torch.save({"mean": mean, "std": std}, standardizer_path)


def load_standardizer(model_dir):
    """Load standardizer parameters from the model directory."""
    standardizer_path = os.path.join(model_dir, "standardizer.pt")
    if os.path.exists(standardizer_path):
        params = torch.load(standardizer_path)
        return params["mean"], params["std"]
    return None, None
