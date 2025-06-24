from abc import ABC, abstractmethod
from typing import Any


class DataSource(ABC):
    """
    Abstract base class for data sources.

    This class defines the interface for loading data from various sources.
    Subclasses should implement the `load` method to provide specific data loading functionality.
    """
    def __call__(self):
        return self.load()

    @abstractmethod
    def load(self):
        raise NotImplementedError("Subclasses should implement this method.")