import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch


class FeatureEncoder(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super(FeatureEncoder, self).__init__()

        self.edge_init = nn.Linear(in_channels, out_channels)

    def forward(self, batch: Batch) -> Batch:

        batch.x = torch.cat([batch.x, batch.extra_atom_features], dim=1)
        row, col = batch.edge_index

        batch.h_0 = F.relu(
            self.edge_init(torch.cat([batch.x[row], batch.edge_attr], dim=1))
        )
        batch.h = batch.h_0

        return batch
