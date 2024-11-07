from abc import ABC, abstractmethod
from typing import List, Optional, Type

from omegaconf import DictConfig


class TransformBase(ABC):
    """Base class for graph transformations."""

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, graph) -> None:
        """Apply transformation to the graph in-place."""
        pass

    @classmethod
    def get_default_config(cls) -> dict:
        """Return default configuration for this transform."""
        return {}
