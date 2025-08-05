from typing import Dict, List, Union

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import degree

from chemtorch.dataset.dataset_base import DatasetBase

class GraphDataset(DatasetBase[Data], Dataset):
    """
    Dataset for molecular graphs.
    It allows for subsampling the data, caching processed graphs, and precomputing all graphs.

    Note:
        This class is designed to work with PyTorch Geometric's Data class and Dataloader.
        It requires a dataframe with a 'label' column and a representation creator that can
        convert the dataframe rows into PyTorch Geometric Data objects.
    """

    @property
    def degree_statistics(self) -> Dict[str, Union[int, List]]:
        """
        Computes degree statistics for the dataset, specifically the overall maximum
        degree and a histogram of node degrees.

        This property requires `precompute_all` to be True during dataset initialization,
        as it iterates over all precomputed graphs.

        Returns:
            Dict[str, Union[int, List]]: A dictionary containing:
                - "max_degree" (int): The maximum degree found across all nodes in all graphs.
                - "degree_histogram" (List[int]) A list where the i-th element
                    is the total count of nodes with degree `i` across all graphs.

        Raises:
            ValueError: If `precompute_all` was False during dataset initialization.
            AttributeError: If the first graph object in the dataset is not a PyTorch Geometric
                            `Data` object or does not possess `edge_index` and `num_nodes` attributes,
                            assuming the dataset is not empty.
        """
        if not self.precompute_all:
            raise ValueError(
                "Dataset must be precomputed to compute degree statistics."
            )

        max_degree = -1
        degree_histogram = None

        if not isinstance(self[0], Data):
            raise AttributeError(
                f"'{self[0].__class__.__name__}' object cannot be used "
                f"to determine degree_statistics"
            )

        for data in self:
            d = degree(
                data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long
            )
            max_degree = max(max_degree, int(d.max()))

            if degree_histogram is None:
                degree_histogram = torch.zeros(
                    max_degree + 1, dtype=torch.long
                )
            elif max_degree >= degree_histogram.numel():
                # Resize the degree_histogram tensor to accommodate the new max_degree
                new_size = max_degree + 1
                resized_histogram = torch.zeros(new_size, dtype=torch.long)
                resized_histogram[: degree_histogram.numel()] = (
                    degree_histogram
                )
                degree_histogram = resized_histogram

            degree_histogram += torch.bincount(
                d, minlength=degree_histogram.numel()
            )
        
        return {
            "max_degree": max_degree,
            "degree_histogram": degree_histogram.tolist(),
        }

    @property
    def num_node_features(self) -> int:
        """
        Returns the number of node features in the dataset. 
        Please refer to the PyTorch Geometric documentation of `torch_geometric.data.Dataset.num_node_features`
        for further details, if necessary.
        """
        return super().num_node_features

    @property
    def num_edge_features(self) -> int:
        """
        Returns the number of edge features in the dataset.
        Please refer to the PyTorch Geometric documentation of `torch_geometric.data.Dataset.num_edge_features`
        for further details, if necessary.
        """
        return super().num_edge_features
