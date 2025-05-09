from abc import ABC, abstractmethod


class DataSource(ABC):
    """
    Abstract base class for data sources.

    This class defines the interface for loading data from various sources.
    Subclasses should implement the `load` method to provide specific data loading functionality.
    """

    @abstractmethod
    def load(self):
        raise NotImplementedError("Subclasses should implement this method.")