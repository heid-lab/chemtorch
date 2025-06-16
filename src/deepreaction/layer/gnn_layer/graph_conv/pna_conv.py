from typing import Dict
from git import List, Union
import torch
import torch.nn as nn
from torch_geometric.nn import PNAConv as pyg_PNAConv


class PNAConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        degree_statistics: Dict[str, Union[int, List[int]]],
        aggregators=["mean", "min", "max", "std"],
        scalers=["identity", "amplification", "attenuation"],
        use_edge_attr: bool = True,
    ):
        super(PNAConv, self).__init__()

        if degree_statistics is None:
            raise ValueError(
                "Dataset degree statistics not found precomputed."
            )
        if "degree_histogram" not in degree_statistics:
            raise ValueError("Degree histogram not found precomputed.")

        deg_hist = degree_statistics["degree_histogram"]
        if not isinstance(deg_hist, torch.Tensor):
            deg_hist = torch.tensor(deg_hist, dtype=torch.long)
        self.degree_histogram = deg_hist

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregators = list(aggregators)
        self.scalers = list(scalers)
        self.use_edge_attr = use_edge_attr
        self.pna_layer = pyg_PNAConv(
            self.in_channels,
            self.out_channels,
            aggregators=self.aggregators,
            scalers=self.scalers,
            deg=self.degree_histogram,
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
