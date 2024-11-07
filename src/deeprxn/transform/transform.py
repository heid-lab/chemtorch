from abc import ABC, abstractmethod
from typing import Any, Dict


class TransformBase(ABC):
    """Base class for molecular graph transforms."""

    def __init__(self, **kwargs):
        """Initialize transform with config parameters."""
        self.config = kwargs

    @abstractmethod
    def __call__(self, data: Any) -> Any:
        """Apply transform to input data.

        Args:
            data: Input data to transform

        Returns:
            Transformed data
        """
        pass

    def __repr__(self) -> str:
        """String representation of transform."""
        return f"{self.__class__.__name__}({self.config})"
