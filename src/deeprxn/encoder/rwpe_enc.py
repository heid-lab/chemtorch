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
        in_channels_pe: int,
    ):
        super().__init__(in_channels, out_channels)

        self.edge_init = nn.Linear(in_channels, out_channels)
        # self.encoder = nn.Linear(in_channels_pe, out_channels)
        self.raw_norm = nn.BatchNorm1d(in_channels_pe * 2)
        self.act = nn.ReLU()
        max_deg = 8

        # self.x_encoder = nn.Linear(88, 300)
        # self.edge_encoder = nn.Linear(22, 300)
        self.deg_encoder = nn.Embedding(max_deg, 88)
        self.rw_encoder = nn.Linear(in_channels_pe * 2, 300)

    def forward(self, batch: Batch) -> Batch:
        row, col = batch.edge_index

        deg = pyg.utils.degree(
            col, num_nodes=batch.num_nodes, dtype=torch.long
        )
        deg_embed = self.deg_encoder(deg)

        # batch.x = self.x_encoder(batch.x)
        batch.x = batch.x + deg_embed

        pos_enc = getattr(batch, "randomwalkpe")
        pos_enc = self.raw_norm(pos_enc)
        pos_enc = self.rw_encoder(pos_enc)
        pos_enc = self.act(pos_enc)
        batch.pos_enc = pos_enc

        # batch.edge_attr = self.edge_encoder(batch.edge_attr)

        batch.h_0 = F.relu(
            self.edge_init(torch.cat([batch.x[row], batch.edge_attr], dim=1))
        )

        batch.h = batch.h_0

        return batch
