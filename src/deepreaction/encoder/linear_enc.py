from torch import nn
from torch_geometric.data import Batch


class LinearEncoder(nn.Module):
    """Linear encoder for graph node and edge features."""

    def __init__(
        self,
        node_in_channels,
        node_out_channels,
        edge_in_channels,
        edge_out_channels,
    ):
        """Initialize the linear encoder.

        Parameters
        ----------
        node_in_channels : int
            The input dimension of node features.
        node_out_channels : int
            The output dimension of node features.
        edge_in_channels : int
            The input dimension of edge features.
        edge_out_channels : int
            The output dimension of edge features.

        """
        super().__init__()

        self.node_encoder = nn.Linear(node_in_channels, node_out_channels)
        self.edge_encoder = nn.Linear(edge_in_channels, edge_out_channels)

    def forward(self, batch):
        batch.x = self.node_encoder(batch.x)
        batch.edge_attr = self.edge_encoder(batch.edge_attr)
        return batch
