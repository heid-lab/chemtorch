from re import sub
import time
from functools import lru_cache
from typing import Callable, List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from deepreaction.dataset import DatasetBase
from deepreaction.representation import AbstractRepresentation
from deepreaction.transform import AbstractTransform


class FingerprintDataset(DatasetBase[torch.Tensor], Dataset):
    """
    Data module for molecular fingerprints.
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
            AbstractRepresentation[torch.Tensor] | Callable[..., torch.Tensor]
        ),
        transform: (
            AbstractTransform[torch.Tensor]
            | Callable[[torch.Tensor], torch.Tensor]
        ) = None,
        precompute_all: bool = True,
        cache: bool = True,
        max_cache_size: Optional[int] = None,
        subsample: Optional[int | float] = None,
    ):
        DatasetBase.__init__(
            self, 
            dataframe=dataframe, 
            representation=representation, 
            transform=transform,
            precompute_all=precompute_all,
            cache=cache,
            max_cache_size=max_cache_size,
            subsample=subsample
        )
        Dataset.__init__(self)


    @property
    def fp_length(self) -> int:
        """Returns the length of fingerprints in the dataset."""
        data = self[0]
        fingerprint = data[0]   # data is a tuple (fingerprint, label)

        if not isinstance(fingerprint, torch.Tensor):
            raise AttributeError(
                f"'{fingerprint.__class__.__name__}' object cannot be used "
                f"to determine fingerprint length"
            )
        
        return fingerprint.shape[0]