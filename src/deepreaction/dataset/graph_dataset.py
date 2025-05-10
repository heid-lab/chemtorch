import time
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import torch
from torch import nn
from torch_geometric.data import Data, Dataset


class GraphDataset(Dataset):
    """
    A flexible dataset class for molecular graphs.

    This class creates graph representations using a provided `representation_creator`
    (e.g., a partial of the CGR class) and then applies any specified `sample_transforms`.
    It supports precomputing all graphs or processing them on demand with caching.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        representation_partial: Callable[..., Data],
        sample_processing_pipeline: nn.Sequential,
        precompute_all: bool = True,
        cache_graphs: bool = True,
        max_cache_size: Optional[int] = None,
        subsample: Optional[int | float] = None,
        *args,  # ingore any additional positional arguments
        **kwargs,  # ignore any additional keyword arguments
    ):
        super().__init__(
            root=None, transform=None, pre_transform=None, pre_filter=None
        )
        """
        Initialize the GraphDataset with the provided data and processing pipeline.

        Args:
            data_df (pd.DataFrame): The input data containing molecular graphs.
            sample_processing_pipeline (nn.Sequential): A pipeline for processing
            individual samples to the final representations on which the model
            will be trained on.
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
        if "label" not in data.columns:
            raise ValueError(
                f"Dataframe must contain a 'label' column, received columns: {data.columns}"
            )

        self.data = self._subsample_data(data, subsample)
        self.representation_partial = representation_partial
        self.sample_processing_pipeline = sample_processing_pipeline

        self.precompute_all = precompute_all
        self.precomputed_graphs: Optional[List[Data]] = None
        self.precompute_time: float = 0.0

        if self.precompute_all:
            print(f"INFO: Precomputing {len(self.data)} graphs...")
            start_time = time.time()
            # Consider using joblib for parallel precomputation if _process_sample_by_idx is slow
            # from joblib import Parallel, delayed
            # self.precomputed_graphs = Parallel(n_jobs=-1)(delayed(self._process_sample_by_idx)(idx) for idx in range(len(self.data_df)))
            self.precomputed_graphs = [
                self._process_sample_by_idx(idx)
                for idx in range(len(self.data))
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

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.data)

    def __getitem__(self, idx):
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

    def get_labels(self):
        """
        Retrieve the labels for the dataset.

        Returns:
            pd.Series: The labels for the dataset.
        """
        return self.data["label"].values

    def _process_sample_by_idx(self, idx) -> Data:
        """
        Process a single sample using this dataset's :attr:`sample_processing_pipeline`.

        Args:
            idx (int): Index of the sample to process.

        Returns:
            Data: A PyTorch Geometric `Data` object representing the molecular graph.

        Raises:
            ValueError: If there is an error during graph creation.
        """
        sample = self.data.iloc[idx]
        try:
            graph_representation = self.representation_partial(**sample)
            return self.sample_processing_pipeline.forward(
                graph_representation
            )
        except Exception as e:
            raise ValueError(f"Error processing sample {idx}, Error: {str(e)}")

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
