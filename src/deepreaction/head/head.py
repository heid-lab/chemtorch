from abc import abstractmethod
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.data import Batch


class Head(nn.Module):
    """Base class for all head implementations."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    @abstractmethod
    def forward(self, batch: Batch) -> torch.Tensor:
        """Process the batch and return predictions.

        Args:
            batch

        Returns:
            torch.Tensor: Output predictions
        """
        pass


class FFNHead(Head):
    """Feed forward network head with configurable layers."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.02,
        activation: str = "relu",
    ):
        super().__init__(in_channels, out_channels)

        # Map activation string to function
        activation = getattr(nn, activation.upper())()

        # Build layers dynamically
        layers = []
        current_dim = in_channels

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Dropout(dropout),
                    nn.Linear(current_dim, hidden_channels),
                    activation,
                ]
            )
            current_dim = hidden_channels

        # Output layer
        layers.extend(
            [nn.Dropout(dropout), nn.Linear(current_dim, out_channels)]
        )

        self.ffn = nn.Sequential(*layers)

    def forward(self, batch: Batch) -> torch.Tensor:
        """Forward pass of the FFN head.

        Args:
            batch: PyG batch with graph embeddings

        Returns:
            torch.Tensor: Output predictions with shape [batch_size, out_channels]
        """
        return self.ffn(batch.x).squeeze(-1)
