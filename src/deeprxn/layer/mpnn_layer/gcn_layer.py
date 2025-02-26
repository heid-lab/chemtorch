import torch_geometric.nn as pyg_nn

from deeprxn.layer.mpnn_layer.mpnn_layer_base import MPNNLayerBase


class GCNLayer(MPNNLayerBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__(in_channels, out_channels)

        self.model = pyg_nn.GCNConv(in_channels, out_channels)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch
