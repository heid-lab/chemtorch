from typing import Any, NamedTuple


class DataSplit(NamedTuple):
    """
    A named tuple to hold the data splits for training, validation, and testing.
    """
    train: Any
    val: Any
    test: Any
    train_coord: Any = None
    val_coord: Any = None
    test_coord: Any = None
