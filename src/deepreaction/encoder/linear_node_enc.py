from torch import nn
from torch_geometric.data import Batch

from deepreaction.encoder.encoder_base import Encoder


class LinearNodeEncoder(Encoder):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
    ):
        super().__init__()

        self.encoder = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, batch: Batch) -> Batch:
        batch.x = self.encoder(batch.x)
        return batch
