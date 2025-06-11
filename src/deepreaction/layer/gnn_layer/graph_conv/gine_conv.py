import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import Linear as pyg_Linear
from torch_geometric.nn import GINEConv as pyg_GINEConv


class GINEConv(nn.Module):
    """Graph Isomorphism Network with Edge features (GINE) layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super(GINEConv, self).__init__()
        gin_nn = nn.Sequential(
            pyg_Linear(in_channels, out_channels),
            nn.ReLU(),
            pyg_Linear(out_channels, out_channels),
        )
        self.model = pyg_GINEConv(gin_nn)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr)
        return batch
