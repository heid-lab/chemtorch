from typing import Callable
from torch import nn
import torch
from torch_geometric.data import Batch

from deepreaction.model.abstract_model import DeepReactionModel


class GNN(nn.Module, DeepReactionModel[Batch]):
    def __init__(
        self,
        encoder: Callable[[Batch], Batch],
        gnn: Callable[[Batch], Batch],
        pool: Callable[[Batch], torch.Tensor],
        head: Callable[[torch.Tensor], torch.Tensor],
    ):
        """
        Args:
            encoder (Callable[[Batch], Batch]): The encoder function that processes the input batch.
            gnn (Callable[[Batch], Batch]): The GNN that processes the batch.
            pool (Callable[[Batch], torch.Tensor]): The pooling function that converts the graph to a tensor.
            head (Callable[[torch.Tensor], torch.Tensor]): The head function for final prediction.
        """
        super().__init__()
        self.encoder = encoder
        self.gnn = gnn
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
        batch = self.gnn(batch)
        batch = self.pool(batch)
        return self.head(batch)

