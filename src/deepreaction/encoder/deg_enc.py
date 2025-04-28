import torch
import torch_geometric as pyg
from torch import nn


class DegreeEncoder(nn.Module):
    """Node degree encoder for graph neural networks."""

    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        """Initialize the degree encoder.

        Parameters
        ----------
        in_channels : int
            The maximum degree to encode (embedding table size).
        out_channels : int
            The dimension of the degree embedding.

        """
        super().__init__()

        self.encoder = nn.Embedding(in_channels, out_channels)

    def forward(self, batch):
        degree = pyg.utils.degree(
            batch.edge_index[1], num_nodes=batch.num_nodes, dtype=torch.float
        )

        degree_emb = self.encoder(degree.type(torch.long))

        batch.x = torch.cat([batch.x, degree_emb], dim=1)

        return batch
