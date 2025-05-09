from torch import nn
from torch_geometric.data import Batch


class LinearNodeEncoder(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
    ):
        super(LinearNodeEncoder, self).__init__()

        self.encoder = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, batch: Batch) -> Batch:
        batch.x = self.encoder(batch.x)
        return batch
