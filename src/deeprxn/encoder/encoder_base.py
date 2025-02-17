from abc import abstractmethod

import torch.nn as nn
from torch_geometric.data import Batch


class Encoder(nn.Module):
    """Base class for all encoders.

    All encoders should inherit from this class and implement the forward method.
    """

    def __init__(self):
        """Initialize the base encoder.

        Args:
        """
        super().__init__()

    @abstractmethod
    def forward(self, batch: Batch) -> Batch:
        """Forward pass of the encoder.

        Args:
            batch: PyG batch containing graph data

        Returns:
            Updated batch with new features
        """
        pass
