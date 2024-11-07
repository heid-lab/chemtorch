from abc import abstractmethod

import torch.nn as nn
from torch_geometric.data import Batch


class Encoder(nn.Module):
    """Base class for all encoders.

    All encoders should inherit from this class and implement the forward method.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """Initialize the base encoder.

        Args:
            in_channels: Number of input features
            out_channels: Number of output features
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    @abstractmethod
    def forward(self, batch: Batch) -> Batch:
        """Forward pass of the encoder.

        Args:
            batch: PyG batch containing graph data

        Returns:
            Updated batch with new features
        """
        pass
