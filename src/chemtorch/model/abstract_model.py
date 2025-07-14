from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import torch


T = TypeVar("T")


class chemtorchModel(ABC, Generic[T]):
    """
    Abstract base class for all models in the chemtorch framework.

    T: The type of the input batch (e.g., `torch.Tensor`, `torch_geometric.data.Batch`, etc.).
    """

    @abstractmethod
    def __call__(self, batch: T) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            batch (T): The input batch of data.

        Returns:
            torch.Tensor: The output predictions.
        """
        pass
