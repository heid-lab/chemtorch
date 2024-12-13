import torch
import torch.nn.functional as F
import torch_geometric as pyg
from torch import nn
from torch_geometric.data import Batch

from deeprxn.encoder.encoder_base import Encoder


class RWEncoder(Encoder):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        # in_channels_edge: int,
        # in_channels_pe: int,
        # out_channels_pe: int,
    ):
        super().__init__()

        # dim_in = in_channels
        dim_pe = in_channels  # * 2  # in_channels_pe * 2
        self.raw_norm = nn.BatchNorm1d(dim_pe)

        # self.linear_x = nn.Linear(dim_in, out_channels - out_channels_pe)
        # self.linear_edge_attr = nn.Linear(in_channels_edge, out_channels)

        self.pe_encoder = nn.Linear(dim_pe, out_channels)  # out_channels_pe

    def forward(self, batch: Batch) -> Batch:

        if not hasattr(batch, "randomwalkpe"):
            raise ValueError(
                "Batch object does not have randomwalkpe attribute"
            )

        pos_enc = getattr(batch, "randomwalkpe")

        pos_enc = self.raw_norm(pos_enc)
        pos_enc = self.pe_encoder(pos_enc)
        # h = self.linear_x(batch.x)
        batch.x = torch.cat([batch.x, pos_enc], dim=1)

        return batch
