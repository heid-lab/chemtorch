from __future__ import annotations
from typing import Any, Generic, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Self

T = TypeVar("T")

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
