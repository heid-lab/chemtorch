import re
import torch

from torch import nn
from torch_geometric.utils import degree
from deeprxn.dataset.graph_dataset import GraphDataset


class DatasetDegreeStatistics(nn.Module):
    """
    A dataset-wide transform to compute degree statistics for PNA.

    This transform computes the maximum degree and degree histogram for the entire dataset.
    """

    def __init__(self):
        super(DatasetDegreeStatistics, self).__init__()
        self.max_degree = -1

    def forward(self, dataset: GraphDataset) -> GraphDataset:
        """
        Compute degree statistics for the entire dataset and saves them as a `dict` 
        under the :args:`dataset.degree_statistics` property.

        Args:
            dataset (GraphDataset): The dataset for which to compute degree statistics.

        Raises:
            TypeError: If the dataset is not an instance of GraphDataset.
            ValueError: If the dataset is not precomputed.
        """
        if not isinstance(dataset, GraphDataset):
            raise TypeError("Dataset must be an instance of GraphDataset.")
        if not dataset.precompute_all:
            raise ValueError("Dataset must be precomputed to compute degree statistics.")

        max_degree = -1
        degree_histogram = None

        for data in dataset:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            max_degree = max(max_degree, int(d.max()))

            if degree_histogram is None:
                degree_histogram = torch.zeros(max_degree + 1, dtype=torch.long)
            elif max_degree >= degree_histogram.numel():
                # Resize the degree_histogram tensor to accommodate the new max_degree
                new_size = max_degree + 1
                resized_histogram = torch.zeros(new_size, dtype=torch.long)
                resized_histogram[:degree_histogram.numel()] = degree_histogram
                degree_histogram = resized_histogram

            degree_histogram += torch.bincount(d, minlength=degree_histogram.numel())

        dataset.degree_statistics = {
            "max_degree": max_degree,
            "degree_histogram": degree_histogram,
        }
        return dataset
