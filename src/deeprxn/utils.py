import os
import random
from pathlib import Path
from typing import Literal, Tuple, Union

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
    use_fraction: bool,    
    split: Literal["train", "val", "test"] = "train",
    data_root: str = "data",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load data from a CSV file with configurable input and target columns.

    Args:
        input_column: Name of the column containing input data (e.g., SMILES strings)
        target_column: Name of the column containing target values
        data_folder: Subfolder name containing the dataset
        split: Which dataset split to load ("train", "val", or "test")
        data_root: Root directory containing all datasets

    Returns:
        Tuple of (inputs, targets) as numpy arrays

    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If required columns are missing
    """
    data_path = Path(data_root) / data_folder / f"{split}.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    if use_fraction and split == "train":
        data_df = pd.read_csv(data_path)
        data_df = data_df.sample(int(len(data_df) * use_fraction)) # select randomly n entries
    else:    
        data_df = pd.read_csv(data_path)

    missing_cols = []
    for col in [input_column, target_column]:
        if col not in data_df.columns:
            missing_cols.append(col)

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    inputs = data_df[input_column].values
    targets = data_df[target_column].values.astype(float)

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


def subset_dataloader(dataloader, fraction):
    if fraction >= 1.0:
        return dataloader

    dataset = dataloader.dataset
    num_samples = int(len(dataset) * fraction)
    indices = torch.randperm(len(dataset))[:num_samples]
    subset = torch.utils.data.Subset(dataset, indices)

    return torch.utils.data.DataLoader(
        subset,
        batch_size=dataloader.batch_size,
        shuffle=True,
        num_workers=dataloader.num_workers,
    )
