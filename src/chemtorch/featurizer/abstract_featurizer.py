from abc import ABC, abstractmethod
from typing import List


class AbstractFeaturizer:
    """
    Abstract class that defines the interface for featurizers insides the
    chemtorch framework.

    This class defines the interface for featurizers that convert input data into a list of numeric features.
    It is designed to be subclassed by specific featurizer implementations.
    """

    @abstractmethod
    def __call__(self, *args, **kwargs) -> List[float | int]:
        """
        Abstract method to be implemented by subclasses.
        This method should take in a sample and return a list of float features.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            List[float]: A list of float features.
        """
        pass
