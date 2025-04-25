from typing import Any, Dict

import torch
from torch_geometric.utils import degree

from deepreaction.transform.transform_base import TransformBase


class BatchedDegreeStatistics(
    TransformBase
):  # TODO: optimize, does inherit make slower?
    def __init__(self, type="dataset"):
        self.max_degree = -1
        self.is_finalized = False
        self.type = type
        self.needs_second_dataloader = True

    def forward(self, batch):
        d = degree(
            batch.edge_index[1],
            num_nodes=batch.num_nodes,
            dtype=torch.long,
        )

        self.max_degree = max(self.max_degree, int(d.max()))

        return batch

    def finalize(self, loader):
        degree_histogram = torch.zeros(self.max_degree + 1, dtype=torch.long)
        for data in loader:
            d = degree(
                data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long
            )
            degree_histogram += torch.bincount(
                d, minlength=degree_histogram.numel()
            )

        return {
            "max_degree": self.max_degree,
            "degree_histogram": degree_histogram,
        }
