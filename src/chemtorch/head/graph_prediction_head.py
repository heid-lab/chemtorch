from typing import Callable
import torch
from torch import nn
from torch_geometric.data import Batch

class GraphPredictionHead(nn.Module):
    def __init__(
            self, 
            batch_to_tensor: Callable,
            predictor: nn.Module, 
        ):
        """
        Args:
            predictor: The prediction head (e.g., MLP, Linear, etc.)
            batch_to_tensor: Callable that extracts the input tensor from the batch.
        """
        super().__init__()
        self.head = predictor
        self.batch_to_tensor = batch_to_tensor

    def forward(self, batch: Batch) -> torch.Tensor:
        x = self.batch_to_tensor(batch)
        return self.head(x)

class BatchToX:
    def __call__(self, batch):
        return batch.x

class BatchToXAndEnthalpy:
    def __call__(self, batch):
        return torch.cat([batch.x, batch.enthalpy.unsqueeze(-1)], dim=1)

