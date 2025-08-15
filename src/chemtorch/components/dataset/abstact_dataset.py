from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Tuple, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    import pandas as pd

T_co = TypeVar("T_co", covariant=True)
R = TypeVar("R")  # Invariant since it's used in both input/output positions

class AbstractDataset(ABC, Generic[T_co, R]):
    """
    Protocol defining the interface for datasets with typed representations.
    
    This allows operations to specify exactly what type of representation
    they expect while maintaining type safety.
    """
    
    representation: R
    dataframe: 'pd.DataFrame'
    has_labels: bool
    precompute_all: bool
    
    @abstractmethod
    def __len__(self) -> int: ...
    
    @abstractmethod
    def __getitem__(self, idx: int) -> T_co | Tuple[T_co, torch.Tensor]: ...

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
