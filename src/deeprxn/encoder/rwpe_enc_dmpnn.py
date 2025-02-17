import torch
import torch.nn.functional as F
import torch_geometric as pyg
from torch import nn
from torch_geometric.data import Batch

from deeprxn.encoder.encoder_base import Encoder


class RWEncoderDMPNN(Encoder):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_channels_pe: int,
    ):
        super().__init__()

        dim_pe = in_channels_pe * 2
        self.raw_norm = nn.BatchNorm1d(dim_pe)
        self.edge_init = nn.Linear(in_channels + dim_pe, out_channels)

    def forward(self, batch: Batch) -> Batch:

        pos_enc = batch.randomwalkpe
        pos_enc = self.raw_norm(pos_enc)
        batch.x = torch.cat([batch.x, pos_enc], dim=1)

        row, col = batch.edge_index
        batch.h_0 = F.relu(
            self.edge_init(torch.cat([batch.x[row], batch.edge_attr], dim=1))
        )
        batch.h = batch.h_0

        return batch
