import time
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Union

import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import degree

from deepreaction.dataset.dataset_base import DatasetBase
from deepreaction.representation.representation_base import RepresentationBase
from deepreaction.transform.abstract_transform import AbstractTransform


class GraphDataset(Dataset, DatasetBase[Data]):
    """
    A flexible dataset class for molecular graphs.
    It allows for subsampling the data, caching processed graphs, and precomputing all graphs.

    Note:
        This class is designed to work with PyTorch Geometric's Data class and Dataloader.
        It requires a dataframe with a 'label' column and a representation creator that can
        convert the dataframe rows into PyTorch Geometric Data objects.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        representation: RepresentationBase[Data] | Callable[..., Data],
        transform: AbstractTransform[Data] | Callable[[Data], Data] = None,
        precompute_all: bool = True,
        cache_graphs: bool = True,
        max_cache_size: Optional[int] = None,
        subsample: Optional[int | float] = None,
        *args,  # ingore any additional positional arguments
        **kwargs,  # ignore any additional keyword arguments
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
            *args: Additional positional arguments, ignored.
            **kwargs: Additional keyword arguments, ignored.

        Raises:
            ValueError: If the data does not contain a 'label' column.
            ValueError: If the subsample is not an int or a float.
            ValueError: If the dataset is not precomputed and caching is not enabled.
        """
        DatasetBase.__init__(self, dataframe, representation, transform)
        Dataset.__init__(self)
        if "label" not in dataframe.columns:
            raise ValueError(
                f"Dataframe must contain a 'label' column, received columns: {dataframe.columns}"
            )

        self.dataframe = self._subsample_data(dataframe, subsample)
        self.representation = representation
        self.transforms = transform

        self.precompute_all = precompute_all
        self.precomputed_graphs: Optional[List[Data]] = None
        self.precompute_time: float = 0.0

        if self.precompute_all:
            print(f"INFO: Precomputing {len(self.dataframe)} graphs...")
            start_time = time.time()
            self.precomputed_graphs = [
                self._process_sample_by_idx(idx)
                for idx in range(len(self.dataframe))
            ]
            self.precompute_time = time.time() - start_time
            print(
                f"INFO: Precomputation finished in {self.precompute_time:.2f}s."
            )
        else:
            if cache_graphs:
                self._get_processed_sample = lru_cache(
                    max_size=max_cache_size
                )(self._process_sample_by_idx)
            else:
                self._get_processed_sample = self._process_sample_by_idx

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.dataframe)

    def __getitem__(self, idx) -> Data:
        """
        Retrieve a processed graph by its index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Data: A PyTorch Geometric `Data` object representing the molecular graph.
        """
        if self.precompute_all:
            if self.precomputed_graphs is None:
                raise RuntimeError(
                    f"Graphs were set to be precomputed but are not available."
                )
            return self.precomputed_graphs[idx]
        else:
            return self._get_processed_sample(idx)

    def _process_sample_by_idx(self, idx: int) -> Data:
        sample = self.dataframe.iloc[idx]
        return self._process_sample(sample)

    # TODO: Remove this method
    def get_labels(self):
        """
        Retrieve the labels for the dataset.

        Returns:
            pd.Series: The labels for the dataset.
        """
        return self.dataframe["label"].values

    def _subsample_data(
        self, data: pd.DataFrame, subsample: Optional[int | float]
    ) -> pd.DataFrame:
        """
        Subsample the data.

        Args:
            data (pd.DataFrame): The original data.
            subsample (Optional[int | float]): The subsample size or fraction.
        Returns:
            pd.DataFrame: The subsampled data.
        """
        if subsample is None or subsample == 1.0:
            return data
        elif isinstance(subsample, int):
            return data.sample(n=subsample)
        elif isinstance(subsample, float):
            return data.sample(frac=subsample)
        else:
            raise ValueError("Subsample must be an int or a float.")

    @property
    def degree_statistics(self) -> Dict[str, Union[int, torch.Tensor]]:
        """
        Computes degree statistics for the dataset, specifically the overall maximum
        degree and a histogram of node degrees.

        This property requires `precompute_all` to be True during dataset initialization,
        as it iterates over all precomputed graphs.

        Returns:
            Dict[str, Union[int, torch.Tensor]]: A dictionary containing:
                - "max_degree" (int): The maximum degree found across all nodes in all graphs.
                - "degree_histogram" (torch.Tensor): A 1D tensor where the i-th element
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
            "degree_histogram": degree_histogram,
        }
