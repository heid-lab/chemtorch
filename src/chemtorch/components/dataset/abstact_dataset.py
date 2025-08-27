from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Tuple, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    import pandas as pd

T = TypeVar("T", covariant=True)
R = TypeVar("R")  # Invariant since it's used in both input/output positions

class AbstractDataset(ABC, Generic[T, R]):
    """
    Protocol defining the interface for datasets with typed representations.
    All datasets in ChemTorch must subclass this protocol and implement the
    required methods. This class can also be used for type hinting where
    `AbstractDataset[T, R]` means that the dataset will produce items of type
    `T` or `Tuple[T, torch.Tensor]` and use a representation of type `R`.

    """
    
    representation: R
    dataframe: 'pd.DataFrame'
    has_labels: bool
    precompute_all: bool
    
    @abstractmethod
    def __len__(self) -> int: ...
    
    @abstractmethod
    def __getitem__(self, idx: int) -> T | Tuple[T, torch.Tensor]: ...

    @abstractmethod   
    def get_labels(self) -> 'torch.Tensor': ...
    
    @property
    @abstractmethod
    def precompute_time(self) -> float: ...
    
    @property 
    @abstractmethod
    def mean(self) -> float: ...
    
    @property
    @abstractmethod
    def std(self) -> float: ...
