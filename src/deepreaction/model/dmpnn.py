from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn.aggr import SumAggregation

from deepreaction.model.abstract_model import DeepReactionModel


class DMPNN(nn.Module, DeepReactionModel[Batch]):
    def __init__(
        self,
        depth: int,
        encoder: Callable[[Batch], Batch],
        layer: Callable[[Batch], Batch],
        edge_to_node_in_channels: int,
        edge_to_node_out_channels: int,
        pool: Callable[[Batch], torch.Tensor],
        head: Callable[[torch.Tensor], torch.Tensor],
    ):
        """
        Args:
            encoder (Callable[[Batch], Batch]): The encoder function that processes the input batch.
            layer (Callable[[Batch], Batch]): The GNN layer that processes the batch.
            pool (Callable[[Batch], torch.Tensor]): The pooling function that converts the graph to a tensor.
            head (Callable[[torch.Tensor], torch.Tensor]): The head function for final prediction.
        """
        super().__init__()
        self.encoder = encoder
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(layer)
        self.aggregation = SumAggregation()

        self.edge_to_node = nn.Linear(
            edge_to_node_in_channels, edge_to_node_out_channels
        )
        self.pool = pool
        self.head = head

    def forward(self, batch: Batch) -> torch.Tensor:
        """
        Forward pass through the GNN model.

        Args:
            batch (Batch): The input batch of graphs.

        Returns:
            torch.Tensor: The output predictions.
        """
        batch = self.encoder(batch)
        for layer in self.layers:
            batch = layer(batch)
        s = self.aggregation(batch.h, batch.edge_index[1])

        batch.q = torch.cat([batch.x, s], dim=1)
        batch.x = F.relu(self.edge_to_node(batch.q))
        batch = self.pool(batch)
        return self.head(batch)
