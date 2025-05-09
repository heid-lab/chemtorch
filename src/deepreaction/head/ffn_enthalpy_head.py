import torch
import torch.nn as nn
from torch_geometric.data import Batch

from deepreaction.act.act import Activation, ActivationType
from deepreaction.head.head import Head


class FFNEnthalpyHead(Head):
    """Feed forward network head with configurable layers."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_layers: int = 2,
        dropout: float = 0.02,
        activation: ActivationType = "relu",
    ):
        super().__init__(in_channels, out_channels)

        # Use our Activation class
        self.activation = Activation(activation_type=activation)

        # Build layers dynamically
        layers = []
        current_dim = in_channels

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Dropout(dropout),
                    nn.Linear(current_dim, hidden_channels),
                    self.activation,
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

        batch.x = torch.cat([batch.x, batch.enthalpy.unsqueeze(-1)], dim=1)

        return self.ffn(batch.x).squeeze(-1)
