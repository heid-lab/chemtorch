from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from deeprxn.layer.mpnn_layer.mpnn_layer_base import MPNNLayerBase


class MPNNLayer(MPNNLayerBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ):
        MPNNLayerBase.__init__(self, in_channels, out_channels)

        self.dropout = dropout
        self.lin_real = nn.Linear(in_channels, out_channels)

    def forward(self, batch: Batch) -> Batch:
        aggregated_messages = self.propagate(batch.edge_index, x=batch.x)
        batch.x = self.lin_real(aggregated_messages)
        batch.x += batch.h_0
        batch.x = F.dropout(
            F.relu(batch.x), self.dropout, training=self.training
        )

        return batch

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        return x_j
