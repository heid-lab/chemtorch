import torch
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv

from deeprxn.layer.mpnn_layer.mpnn_layer_base import MPNNLayerBase


class GATLayer(MPNNLayerBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        dropout: float = 0.0,
        concat: bool = True,
    ):
        super().__init__(in_channels, out_channels)

        self.gat = GATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout,
            concat=concat,
        )

    def forward(self, batch: Batch) -> Batch:
        batch.x = self.gat(batch.x, batch.edge_index, batch.edge_attr)
        return batch
