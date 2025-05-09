from torch import nn
from torch_geometric.data import Batch


class LinearEdgeEncoder(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
    ):
        super(LinearEdgeEncoder).__init__()

        self.encoder = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, batch: Batch) -> Batch:
        batch.edge_attr = self.encoder(batch.edge_attr)
        return batch
