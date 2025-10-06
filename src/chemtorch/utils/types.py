from __future__ import annotations

import torch
import pandas as pd

from typing import Any, Callable, Generic, List, Literal, Optional, TypeVar, TYPE_CHECKING, Union
from typing_extensions import Protocol

if TYPE_CHECKING:
    # Importing here to avoid circular imports
    from torch.utils.data import DataLoader
    from chemtorch.core.dataset_base import DatasetBase
    from chemtorch.components.transform.abstract_transform import AbstractTransform
    from chemtorch.components.augmentation.abstract_augmentation import AbstractAugmentation
    from chemtorch.utils.callable_compose import CallableCompose

T = TypeVar("T")

LightningTask = Literal["fit", "validate", "test", "predict"]
DatasetKey = Literal["train", "val", "test", "predict"]
PropertySource = Literal["any", "all", "train", "val", "test", "predict"]

if TYPE_CHECKING:
    TransformType = Union[AbstractTransform[T], CallableCompose[T, T], Callable[[T], T]]
    AugmentationType = Union[AbstractAugmentation[T], Callable[[T, Optional[torch.Tensor]], List[tuple[T, Optional[torch.Tensor]]]]]
else:
    # Runtime fallbacks
    TransformType = Union[Any, Callable[[Any], Any]]
    AugmentationType = Union[Any, Callable[[Any, Optional[torch.Tensor]], List[tuple[Any, Optional[torch.Tensor]]]]]


class DataLoaderFactoryProtocol(Protocol):
    """Protocol defining the interface for dataloader factory functions."""
    
    def __call__(
        self,
        *,
        dataset: Any,  # Use Any at runtime
        shuffle: bool,
        **kwargs: Any
    ) -> DataLoader:  # Use Any at runtime
        """
        Create a DataLoader from a dataset.
        
        Args:
            dataset: The dataset to create a dataloader for.
            shuffle: Whether to shuffle the data.
            **kwargs: Additional keyword arguments passed to the DataLoader constructor.
            
        Returns:
            A DataLoader instance.
        """
        ...

class DataSplit(Generic[T]):
    """
    A data structure to hold the data splits for training, validation, and testing.
    Provides named tuple-like access with generic typing support.
    """
    
    def __init__(self, train: T, val: T, test: T) -> None:
        self.train = train
        self.val = val
        self.test = test
    
    def __repr__(self) -> str:
        return f"DataSplit(train={self.train!r}, val={self.val!r}, test={self.test!r})"
    
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DataSplit):
            return False
        return (
            self.train == other.train and
            self.val == other.val and
            self.test == other.test
        )
    
    def __iter__(self):
        """Allow tuple unpacking: train, val, test = data_split"""
        yield self.train
        yield self.val  
        yield self.test
    
    def __getitem__(self, index: int) -> T:
        """Allow indexing: data_split[0] == train"""
        if index == 0:
            return self.train
        elif index == 1:
            return self.val
        elif index == 2:
            return self.test
        else:
            raise IndexError("DataSplit index out of range")
    
    def to_dict(self) -> dict[str, T]:
        return {
            "train": self.train,
            "val": self.val,
            "test": self.test,
        }