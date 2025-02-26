from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from deeprxn.layer.mpnn_layer.mpnn_layer_base import MPNNLayerBase


class DMPNNLayer(MPNNLayerBase):
    """Directed Message Passing Neural Network Layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        separate_nn: bool = False,
    ):
        """Initialize DMPNN layer.

        Args:
            in_channels: Number of input features
            out_channels: Number of output features
            separate_nn: Whether to use separate neural networks for real and artificial bonds
        """
        MPNNLayerBase.__init__(self, in_channels, out_channels)

        self.separate_nn = separate_nn
        self.lin_real = nn.Linear(in_channels, out_channels)
        if separate_nn:
            self.lin_artificial = (
                nn.Linear(in_channels, out_channels)
                if separate_nn
                else self.lin_real
            )

    def forward(self, batch: Batch) -> Batch:
        """Forward pass of the DMPNNLayer.

        Args:
            batch: PyG batch containing:
                - edge_index: Tensor of shape [2, num_edges]
                - edge_attr: Tensor of shape [num_edges, in_channels]
                - is_real_bond: Optional boolean tensor of shape [num_edges]

        Returns:
            Updated batch with new edge attributes
        """
        edge_index = batch.edge_index
        is_real_bond = getattr(batch, "is_real_bond", None)

        row, col = edge_index

        aggregated_messages = self.propagate(edge_index, edge_attr=batch.h)
        rev_messages = self._compute_reverse_messages(batch.h)

        if self.separate_nn and is_real_bond is not None:
            batch.h = torch.where(
                is_real_bond.unsqueeze(1),
                self.lin_real(aggregated_messages[row] - rev_messages),
                self.lin_artificial(aggregated_messages[row] - rev_messages),
            )
        else:
            batch.h = self.lin_real(aggregated_messages[row] - rev_messages)

        return batch

    def message(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """Message function for the DMPNNLayer.

        Args:
            edge_attr: Edge features

        Returns:
            Messages to be aggregated
        """
        return edge_attr

    def _compute_reverse_messages(
        self, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """Compute reverse messages for each edge.

        Args:
            edge_attr: Edge features

        Returns:
            Reversed messages
        """
        try:
            return torch.flip(
                edge_attr.view(edge_attr.size(0) // 2, 2, -1), dims=[1]
            ).view(edge_attr.size(0), -1)
        except:
            print("Error in _compute_reverse_messages")
            return torch.zeros_like(edge_attr)
