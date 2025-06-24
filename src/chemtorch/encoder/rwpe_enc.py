import torch
import torch.nn.functional as F
import torch_geometric as pyg
from torch import nn
from torch_geometric.data import Batch


class RWEncoder(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        as_variable: bool = False,
    ):
        super(RWEncoder, self).__init__()
        self.as_variable = as_variable
        self.raw_norm = nn.BatchNorm1d(in_channels)
        self.pe_encoder = nn.Linear(in_channels, out_channels)

    def forward(self, batch: Batch) -> Batch:

        if not hasattr(batch, "randomwalkpe"):
            raise ValueError(
                "Batch object does not have randomwalkpe attribute"
            )

        pos_enc = getattr(batch, "randomwalkpe")

        pos_enc = self.raw_norm(pos_enc)
        pos_enc = self.pe_encoder(pos_enc)

        if self.as_variable:
            batch.randomwalkpe = pos_enc
        else:
            batch.x = torch.cat([batch.x, pos_enc], dim=1)

        return batch
