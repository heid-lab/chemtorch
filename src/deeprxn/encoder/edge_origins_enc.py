import torch
from torch import nn
from torch_geometric.data import Batch

from deeprxn.encoder.encoder_base import Encoder


class EdgeOriginsEncoder(Encoder):

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

        if not hasattr(batch, "edge_origin_encoding"):
            raise ValueError(
                "Batch object does not have edge_origin_encoding attribute"
            )

        edge_origin_encoding = getattr(batch, "edge_origin_encoding")

        edge_origin_encoding = self.raw_norm(edge_origin_encoding)
        edge_origin_encoding = self.linear(edge_origin_encoding)

        if self.as_variable:
            batch.edge_origin_encoding = edge_origin_encoding
        else:
            batch.edge_attr = torch.cat(
                [batch.edge_attr, edge_origin_encoding], dim=1
            )

        return batch
