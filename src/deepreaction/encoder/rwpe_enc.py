import torch
import torch.nn.functional as F
import torch_geometric as pyg
from torch import nn
from torch_geometric.data import Batch

from deepreaction.encoder.encoder_base import Encoder


class RWEncoder(Encoder):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        transform_type: str = "normal",
        as_variable: bool = False,
    ):
        super().__init__()
        self.as_variable = as_variable

        in_channels = (
            in_channels // 2 if transform_type == "difference" else in_channels
        )

        self.raw_norm = nn.BatchNorm1d(in_channels)
        self.transform_type = transform_type

        self.pe_encoder = nn.Linear(in_channels, out_channels)

    def forward(self, batch: Batch) -> Batch:

        if not hasattr(batch, "randomwalkpe"):
            raise ValueError(
                "Batch object does not have randomwalkpe attribute"
            )

        pos_enc = getattr(batch, "randomwalkpe")

        if self.transform_type == "reactant_difference":
            mid = pos_enc.shape[-1] // 2
            reactant, product = pos_enc[..., :mid], pos_enc[..., mid:]
            pos_enc = torch.cat([reactant, product - reactant], dim=-1)
        if self.transform_type == "difference":
            mid = pos_enc.shape[-1] // 2
            reactant, product = pos_enc[..., :mid], pos_enc[..., mid:]
            pos_enc = product - reactant

        pos_enc = self.raw_norm(pos_enc)
        pos_enc = self.pe_encoder(pos_enc)
        # h = self.linear_x(batch.x)
        if self.as_variable:
            batch.randomwalkpe = pos_enc
        else:
            batch.x = torch.cat([batch.x, pos_enc], dim=1)

        return batch
