import torch
import torch_geometric as pyg
from torch import nn
from torch_geometric.data import Batch

from deepreaction.encoder.encoder_base import Encoder


class DegreeEncoder(Encoder):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.encoder = nn.Embedding(in_channels, out_channels)

    def forward(self, batch: Batch) -> Batch:

        degree = pyg.utils.degree(
            batch.edge_index[1], num_nodes=batch.num_nodes, dtype=torch.float
        )

        degree_emb = self.encoder(degree.type(torch.long))

        batch.x = torch.cat([batch.x, degree_emb], dim=1)

        return batch
