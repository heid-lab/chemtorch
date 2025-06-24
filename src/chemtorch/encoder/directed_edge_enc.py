from re import M
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch


class DirectedEdgeEncoder(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super(DirectedEdgeEncoder, self).__init__()

        self.edge_init = nn.Linear(in_channels, out_channels)

    def forward(self, batch: Batch) -> Batch:
        row, col = batch.edge_index
        batch.h_0 = F.relu(
            self.edge_init(torch.cat([batch.x[row], batch.edge_attr], dim=1))
        )
        batch.h = batch.h_0

        return batch
