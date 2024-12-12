import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import Linear as Linear_pyg

from deeprxn.layer.mpnn_layer.mpnn_layer_base import MPNNLayer


class GINELayer(MPNNLayer):
    """Graph Isomorphism Network with Edge features (GINE) layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__(in_channels, out_channels)
        gin_nn = nn.Sequential(
            Linear_pyg(in_channels, out_channels),
            nn.ReLU(),
            Linear_pyg(out_channels, out_channels),
        )
        self.model = pyg_nn.GINEConv(gin_nn)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr)
        # TODO: look into this
        # batch.x = F.relu(batch.x)
        # batch.x = F.dropout(batch.x, p=self.dropout, training=self.training)

        # if self.residual:
        #     batch.x = x_in + batch.x  # residual connection
        return batch
