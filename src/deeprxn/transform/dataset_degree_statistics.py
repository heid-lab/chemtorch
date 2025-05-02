from typing import Any, Dict
import torch
from torch_geometric.utils import degree
from deeprxn.data_pipeline.data_pipeline import DataPipelineComponent


class DatasetDegreeStatistics(DataPipelineComponent):
    """
    A dataset-wide transform to compute degree statistics for PNA.

    This transform computes the maximum degree and degree histogram for the entire dataset.
    """

    def __init__(self):
        self.max_degree = -1

    def forward(self, dataset) -> None:
        """
        Compute degree statistics for the entire dataset and saves them as a `dict` 
        under the :args:`dataset.degree_statistics` property.

        Args:
            dataset: The dataset to process.
        """
        degree_histogram = None

        for data in dataset:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            max_degree = max(max_degree, int(d.max()))

            if degree_histogram is None:
                degree_histogram = torch.zeros(max_degree + 1, dtype=torch.long)

            degree_histogram += torch.bincount(d, minlength=degree_histogram.numel())

        dataset.degree_statistics = {
            "max_degree": max_degree,
            "degree_histogram": degree_histogram,
        }
