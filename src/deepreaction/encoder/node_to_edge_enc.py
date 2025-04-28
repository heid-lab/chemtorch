import torch
import torch.nn.functional as F
from torch import nn


class NodeToEdgeEncoder(nn.Module):
    """Node feature to edge feature encoder for graph neural networks."""

    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        """Initialize the node-to-edge encoder.

        Parameters
        ----------
        in_channels : int
            The dimension of input features (node features + edge attributes).
        out_channels : int
            The dimension of output edge features.

        """
        super().__init__()

        self.edge_init = nn.Linear(in_channels, out_channels)

    def forward(self, batch):
        row, col = batch.edge_index
        batch.h_0 = F.relu(
            self.edge_init(torch.cat([batch.x[row], batch.edge_attr], dim=1))
        )
        batch.h = batch.h_0

        return batch
