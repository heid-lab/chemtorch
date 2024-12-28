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

        if not hasattr(batch, "node_origin_encoding"):
            raise ValueError(
                "Batch object does not have node_origin_encoding attribute"
            )

        node_origin_encoding = getattr(batch, "node_origin_encoding")

        node_origin_encoding = self.raw_norm(node_origin_encoding)
        node_origin_encoding = self.linear(node_origin_encoding)

        if self.as_variable:
            batch.node_origin_encoding = node_origin_encoding
        else:
            batch.x = torch.cat([batch.x, node_origin_encoding], dim=1)

        return batch
