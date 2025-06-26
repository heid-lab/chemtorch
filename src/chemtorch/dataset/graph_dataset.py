import time
from functools import lru_cache
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import degree

from chemtorch.dataset.dataset_base import DatasetBase
from chemtorch.representation import AbstractRepresentation
from chemtorch.transform import AbstractTransform

T = TypeVar("T")

# TODO: Centralize logging
# TODO: Consider saving the precomputed data objects to disk to
# save preprocessing time for repeated runs with the same dataset.
# Note: Update precompute_time property to return 0 or time taken
# to load from disk.
# TODO: Generalize to unlabeled datasets.


class GraphDataset(DatasetBase[Data], Dataset):
    """
    A dataset for molecular graphs, compatible with PyTorch Geometric.

    This class loads data from a pandas DataFrame, converts each row into a
    PyTorch Geometric `Data` object using a provided representation, and applies
    optional transforms. It supports in-memory precomputation, caching, and
    subsampling for efficient data handling.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        representation: Union[
            AbstractRepresentation[Data], Callable[..., Data]
        ],
        transform: Optional[AbstractTransform[T] | Callable[[T], T]] = None,
        precompute_all: bool = True,
        cache: bool = True,
        max_cache_size: Optional[int] = None,
        subsample: Optional[int | float] = None,
    ):
        """
        Initialize the GraphDataset.

        Args:
            dataframe (pd.DataFrame): The input data. Each row must contain a 'label' column.
            representation: A callable that constructs a PyG `Data` object from a dataframe row.
            transform: An optional transformation to apply to each `Data` object.
            precompute_all (bool): If True, process and store all samples in memory on init.
            cache (bool): If True and `precompute_all` is False, cache processed samples.
            max_cache_size (Optional[int]): Maximum size of the LRU cache.
            subsample: The number (int) or fraction (float) of samples to use.
        """

        super().__init__(root=None, transform=None, pre_transform=None)

        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("dataframe must be a pandas DataFrame.")
        if "label" not in dataframe.columns:
            raise ValueError(
                f"Dataframe must contain a 'label' column, received columns: {dataframe.columns}"
            )
        if not callable(representation):
            raise ValueError("representation must be a callable.")
        if not (callable(transform) or transform is None):
            raise ValueError("transform must be a callable or None.")

        self.dataframe = self._subsample_data(dataframe, subsample)
        self.representation = representation
        self.transform = transform

        self.precompute_all = precompute_all
        self.precomputed_items = None
        self._precompute_time = 0.0

        if self.precompute_all:
            print(f"INFO: Precomputing {len(self.dataframe)} items...")
            start_time = time.time()
            self.precomputed_items = [
                self._process_sample(idx) for idx in range(len(self.dataframe))
            ]
            self._precompute_time = time.time() - start_time
            print(
                f"INFO: Precomputation finished in {self._precompute_time:.2f}s."
            )
        else:
            if cache:
                self.process_sample = lru_cache(max_size=max_cache_size)(
                    self._process_sample
                )
            else:
                self.process_sample = self._process_sample

        self._initialized_by_base = True  # mark successful call

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Tuple[Data, torch.Tensor]:
        """
        Retrieve a processed item by its index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            Data: A PyTorch Geometric `Data` object representing the molecular graph.
        """
        if self.precompute_all:
            if self.precomputed_items is None:
                raise RuntimeError("Precomputed items are not available.")
            return self.precomputed_items[idx]
        else:
            return self.process_sample(idx)

    def get_labels(self) -> pd.Series:
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

    def _process_sample(self, idx: int) -> Tuple[T, torch.Tensor]:
        """
        Process a sample by its index.

        This method uses the representation callable to create a representation data
        object from the sample data. If a transform is provided, it applies the transform
        to the representation object.

        Args:
            idx (int): The index of the sample to process.

        Returns:
            Tuple[T, torch.Tensor]: A tuple containing the processed data object and its label.

        Raises:
            RuntimeError: If there is an error processing the sample at the given index.
        """
        try:
            row = self.dataframe.iloc[idx]
            label = torch.tensor(row["label"], dtype=torch.float)
            sample = row.drop("label")
            data_obj = self.representation(**sample)
            if self.transform:
                data_obj = self.transform(data_obj)
        except Exception as e:
            raise RuntimeError(f"Error processing sample at index {idx}: {e}")
        return data_obj, label

    @property
    def precompute_time(self) -> float:
        """
        Get the time taken to precompute all samples.

        Returns:
            float: The time in seconds taken to precompute all samples.
        """
        if not self.precompute_all:
            raise RuntimeError(
                "Precomputation is not enabled for this dataset."
            )
        return self._precompute_time

    @property
    def mean(self) -> float:
        """
        Get the mean of the labels in the dataset.

        Returns:
            float: The mean of the labels.
        """
        return np.mean(self.dataframe["label"].values).item()

    @property
    def std(self) -> float:
        """
        Get the standard deviation of the labels in the dataset.

        Returns:
            float: The standard deviation of the labels.
        """
        return np.std(self.dataframe["label"].values).item()

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

        if not isinstance(self[0][0], Data):
            raise AttributeError(
                f"'{self[0][0].__class__.__name__}' object cannot be used "
                f"to determine degree_statistics"
            )

        for data in self:
            data = data[0]
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
