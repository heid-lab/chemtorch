from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv as pyg_GATConv


class GATConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        dropout: float = 0.0,
        concat: bool = True,
        use_edge_attr: bool = True,
    ):
        super(GATConv, self).__init__()
        self.use_edge_attr = use_edge_attr

        self.gat = pyg_GATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout,
            concat=concat,
            edge_dim=in_channels if use_edge_attr else None,
        )

    def forward(self, batch: Batch) -> Batch:
        if self.use_edge_attr:
            batch.x = self.gat(batch.x, batch.edge_index, batch.edge_attr)
        else:
            batch.x = self.gat(batch.x, batch.edge_index)
        return batch
