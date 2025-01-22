import torch
from torch import nn
from torch_geometric.data import Batch

from deeprxn.encoder.encoder_base import Encoder


class NodeOriginsEncoder(Encoder):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        as_variable: bool = False,
    ):
        super().__init__()
        self.as_variable = as_variable
        self.raw_norm = nn.BatchNorm1d(in_channels)
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, batch: Batch) -> Batch:

        if not hasattr(batch, "atom_origin_type"):
            raise ValueError(
                "Batch object does not have atom_origin_type attribute"
            )

        atom_origin_type = getattr(batch, "atom_origin_type")

        one_hot = torch.zeros(
            (atom_origin_type.size(0), 2),
            device=atom_origin_type.device,
            dtype=torch.float,
        )
        one_hot.scatter_(1, atom_origin_type.unsqueeze(1), 1)

        encoded = self.raw_norm(one_hot)
        encoded = self.linear(encoded)

        if self.as_variable:
            batch.atom_origin_type_encoded = encoded
        else:
            batch.x = torch.cat([batch.x, encoded], dim=1)

        return batch
