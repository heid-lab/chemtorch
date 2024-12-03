from torch import nn
from torch_geometric.data import Batch

from deeprxn.encoder.encoder_base import Encoder


class LinearEncoder(Encoder):

    def __init__(
        self,
        node_in_channels: int,
        node_out_channels: int,
        edge_in_channels: int,
        edge_out_channels: int,
    ):
        super().__init__()

        self.node_encoder = nn.Linear(node_in_channels, node_out_channels)
        self.edge_encoder = nn.Linear(edge_in_channels, edge_out_channels)

    def forward(self, batch: Batch) -> Batch:
        batch.x = self.node_encoder(batch.x)
        batch.edge_attr = self.edge_encoder(batch.edge_attr)
        return batch
