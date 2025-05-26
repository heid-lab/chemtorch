from typing import Callable, Dict, List, Optional, Union

import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import degree

from deepreaction.dataset.dataset_base import DatasetBase
from deepreaction.representation import AbstractRepresentation
from deepreaction.transform import AbstractTransform


# TODO: Switching the order of inheritance breaks initialization because
# torch_geometric's Dataset class `super().__init__()` which resolves 
# to the next class in the MRO (DataModuleBase), not and the the parent
# class of Dataset as intended, causing an error because DataModuleBase
# does not receive its expected arguments.
# TODO: Consider using `torch_geometric.data.Dataset.download()` to save the precomputed
# graphs to disk to save preprocessing time in the future.
class GraphDataset(DatasetBase[Data], Dataset):
    """
    Data module for molecular graphs.
    It allows for subsampling the data, caching processed graphs, and precomputing all graphs.

    Note:
        This class is designed to work with PyTorch Geometric's Data class and Dataloader.
        It requires a dataframe with a 'label' column and a representation creator that can
        convert the dataframe rows into PyTorch Geometric Data objects.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        representation: AbstractRepresentation[Data] | Callable[..., Data],
        transform: AbstractTransform[Data] | Callable[[Data], Data] = None,
        precompute_all: bool = True,
        cache: bool = True,
        max_cache_size: Optional[int] = None,
        subsample: Optional[int | float] = None,
    ):
        """
        Initialize the GraphDataset.

        Args:
            dataframe (pd.DataFrame): The dataframe containing the input data.
            representation (RepresentationBase[Data] | Callable[..., Data]): The representation creator.
            transform (TransformBase[Data] | Callable[[Data], Data]): The transform to apply to the data.
            precompute_all (bool): Whether to precompute all graphs upfront.
            cache_graphs (bool): Whether to cache processed graphs (if not precomputing).
            max_cache_size (Optional[int]): Maximum size of the cache (if caching is enabled).
            subsample (Optional[int | float]): The subsample size or fraction.

        Raises:
            ValueError: If the data does not contain a 'label' column.
            ValueError: If the subsample is not an int or a float.
            ValueError: If the dataset is not precomputed and caching is not enabled.
        """
        DatasetBase.__init__(
            self,
            dataframe=dataframe, 
            representation=representation, 
            transform=transform,
            precompute_all=precompute_all,
            cache=cache,
            max_cache_size=max_cache_size,
            subsample=subsample
        )
        Dataset.__init__(self)

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
