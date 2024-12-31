import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import Linear as Linear_pyg

from deeprxn.layer.mpnn_layer.mpnn_layer_base import MPNNLayer


class PNALayer(MPNNLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dataset_precomputed: dict,
        aggregators=["mean", "min", "max", "std"],
        scalers=["identity", "amplification", "attenuation"],
    ):
        super().__init__(in_channels, out_channels)

        if "degree_histogram" not in dataset_precomputed:
            raise ValueError("Degree histogram not found precomputed.")

        self.deg = dataset_precomputed["degree_histogram"]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregators = list(aggregators)
        self.scalers = list(scalers)
        self.pna_layer = pyg_nn.PNAConv(
            in_channels,
            out_channels,
            aggregators=aggregators,
            scalers=scalers,
            deg=self.deg,
            edge_dim=out_channels,
            towers=1,  # TODO: look into this
            pre_layers=1,
            post_layers=1,
            divide_input=False,
        )

    def forward(self, batch):
        batch.x = self.pna_layer(batch.x, batch.edge_index, batch.edge_attr)
        return batch
