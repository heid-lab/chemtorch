import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import Linear as Linear_pyg

from deepreaction.layer.mpnn_layer.mpnn_layer_base import MPNNLayerBase


class PNALayer(MPNNLayerBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dataset_precomputed: dict,
        aggregators=["mean", "min", "max", "std"],
        scalers=["identity", "amplification", "attenuation"],
        use_edge_attr: bool = True,
    ):
        super().__init__(in_channels, out_channels)

        if "degree_histogram" not in dataset_precomputed:
            raise ValueError("Degree histogram not found precomputed.")

        self.deg = dataset_precomputed["degree_histogram"]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregators = list(aggregators)
        self.scalers = list(scalers)
        self.use_edge_attr = use_edge_attr
        self.pna_layer = pyg_nn.PNAConv(
            self.in_channels,
            self.out_channels,
            aggregators=self.aggregators,
            scalers=self.scalers,
            deg=self.deg,
            edge_dim=self.in_channels if use_edge_attr else None,
            towers=1,
            pre_layers=1,
            post_layers=1,
            divide_input=False,
        )

    def forward(self, batch):
        if self.use_edge_attr:
            batch.x = self.pna_layer(
                batch.x, batch.edge_index, batch.edge_attr
            )
        else:
            batch.x = self.pna_layer(batch.x, batch.edge_index)
        return batch
