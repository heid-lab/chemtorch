from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")


class DatasetBase(ABC, Generic[T]):
    """
    Abstract base class for all datasets in the chemtorch framework.

    This class defines the standard interface for all dataset
    implementations. It ensures that different dataset types can be used
    interchangeably by other components of the framework, such as trainers and
    data loaders.

    Subclasses must implement the `__getitem__` and `__len__` methods.

    T: The type of the item returned by `__getitem__`. This can be a single
       tensor, a tuple of tensors, a dictionary, or any other data structure
       that represents a single training example.
    """

    @abstractmethod
    def __getitem__(self, idx: int) -> T:
        """
        Retrieves the data sample corresponding to the given index.

        This method is the core of the dataset, responsible for loading,
        processing, and returning a single data point on demand. The returned
        sample can be a single object, a tuple (e.g., features and labels),
        or any other structure that a model or training loop can consume.

        Args:
            idx (int): The index of the sample to retrieve, in the range
                `[0, len(dataset) - 1]`.

        Returns:
            T: The data sample at the specified index.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        pass
