from typing import Any, Generic, NamedTuple, TypeVar


T = TypeVar("T")

class DataSplit(NamedTuple, Generic[T]):
    """
    A named tuple to hold the data splits for training, validation, and testing.
    """
    train: T
    val: T
    test: T

    def to_dict(self):
        return {
            "train": self.train,
            "val": self.val,
            "test": self.test,
        }
