import time
from functools import lru_cache
from typing import Callable, List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from deepreaction.dataset.dataset_base import DatasetBase
from deepreaction.representation.representation_base import RepresentationBase
from deepreaction.transform.transform_base import TransformBase


class FingerprintDataset(DatasetBase[torch.Tensor], Dataset):
    """
    A flexible dataset class for molecular fingerprints.
    It allows for subsampling the data, caching processed fingerprints, and precomputing all fingerprints.

    Note:
        This class is designed to work with PyTorch's Tensor class and Dataloader.
        It requires a dataframe with a 'label' column and a representation creator that can
        convert the dataframe rows into PyTorch Tensor objects.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        representation: (
            RepresentationBase[torch.Tensor] | Callable[..., torch.Tensor]
        ),
        transform: (
            TransformBase[torch.Tensor]
            | Callable[[torch.Tensor], torch.Tensor]
        ) = None,
        precompute_all: bool = True,
        cache_fingerprints: bool = True,
        max_cache_size: Optional[int] = None,
        subsample: Optional[int | float] = None,
        *args,  # ingore any additional positional arguments
        **kwargs,  # ignore any additional keyword arguments
    ):
        """
        Initialize the FingerprintDataset.

        Args:
            dataframe (pd.DataFrame): The dataframe containing the input data.
            representation (RepresentationBase[torch.Tensor] | Callable[..., torch.Tensor]): The representation creator.
            transform (TransformBase[torch.Tensor] | Callable[[torch.Tensor], torch.Tensor]): The transform to apply to the data.
            precompute_all (bool): Whether to precompute all fingerprints upfront.
            cache_fingerprints (bool): Whether to cache processed fingerprints (if not precomputing).
            max_cache_size (Optional[int]): Maximum size of the cache (if caching is enabled).
            subsample (Optional[int | float]): The subsample size or fraction.
            *args: Additional positional arguments, ignored.
            **kwargs: Additional keyword arguments, ignored.

        Raises:
            ValueError: If the data does not contain a 'label' column.
            ValueError: If the subsample is not an int or a float.
            ValueError: If the dataset is not precomputed and caching is not enabled.
        """
        DatasetBase.__init__(self, dataframe, representation, transform)
        Dataset.__init__(self)
        if "label" not in dataframe.columns:
            raise ValueError(
                f"Dataframe must contain a 'label' column, received columns: {dataframe.columns}"
            )

        self.dataframe = self._subsample_data(dataframe, subsample)
        self.representation = representation
        self.transforms = transform

        self.precompute_all = precompute_all
        self.precomputed_fingerprints: Optional[List[torch.Tensor]] = None
        self.precompute_time: float = 0.0

        if self.precompute_all:
            print(f"INFO: Precomputing {len(self.dataframe)} fingerprints...")
            start_time = time.time()
            self.precomputed_fingerprints = [
                self._process_sample_by_idx(idx)
                for idx in range(len(self.dataframe))
            ]
            self.precompute_time = time.time() - start_time
            print(
                f"INFO: Precomputation finished in {self.precompute_time:.2f}s."
            )
        else:
            if cache_fingerprints:
                self._get_processed_sample = lru_cache(
                    max_size=max_cache_size
                )(self._process_sample_by_idx)
            else:
                self._get_processed_sample = self._process_sample_by_idx

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.dataframe)

    def __getitem__(self, idx) -> torch.Tensor:
        """
        Retrieve a processed fingerprint by its index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Data: A PyTorch `Tensor` object representing the molecular fingerprint.
        """
        if self.precompute_all:
            if self.precomputed_fingerprints is None:
                raise RuntimeError(
                    f"Fingerprints were set to be precomputed but are not available."
                )
            return self.precomputed_fingerprints[idx]
        else:
            return self._get_processed_sample(idx)

    def _process_sample_by_idx(self, idx: int) -> torch.Tensor:
        sample = self.dataframe.iloc[idx]
        X = self.representation(sample["smiles"])
        return X, torch.tensor(sample["label"], dtype=torch.float32)

    @property
    def fp_length(self) -> int:
        """Returns the length of fingerprints in the dataset."""
        data = self[0]
        fingerprint = data[0] if isinstance(data, tuple) else data

        if isinstance(fingerprint, torch.Tensor):
            return fingerprint.shape[0]

        raise AttributeError(
            f"'{fingerprint.__class__.__name__}' object cannot be used "
            f"to determine fingerprint length"
        )

    # TODO: Remove this method
    def get_labels(self):
        """
        Retrieve the labels for the dataset.

        Returns:
            pd.Series: The labels for the dataset.
        """
        return self.dataframe["label"].values

    def _subsample_data(
        self, data: pd.DataFrame, subsample: Optional[int | float]
    ) -> pd.DataFrame:
        """
        Subsample the data.

        Args:
            data (pd.DataFrame): The original data.
            subsample (Optional[int | float]): The subsample size or fraction.
        Returns:
            pd.DataFrame: The subsampled data.
        """
        if subsample is None or subsample == 1.0:
            return data
        elif isinstance(subsample, int):
            return data.sample(n=subsample)
        elif isinstance(subsample, float):
            return data.sample(frac=subsample)
        else:
            raise ValueError("Subsample must be an int or a float.")
