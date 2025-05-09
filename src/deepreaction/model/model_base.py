from abc import abstractmethod

import torch.nn as nn
from torch_geometric.data import Batch


class Model(nn.Module):
    """Base class for all graph neural network models."""

    def __init__(
        self,
    ):
        """Initialize base model.

        Args:
            in_channels: Number of input features
            hidden_channels: Number of hidden features
            out_channels: Number of output features
            num_layers: Number of message passing layers
        """
        super().__init__()

    @abstractmethod
    def forward(self, batch: Batch) -> Batch:
        """Forward pass of the model.

        Args:
            batch: PyG batch containing graph data

        Returns:
            Batch with predictions
        """
        pass
