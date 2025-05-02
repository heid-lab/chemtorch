from functools import lru_cache
from typing import Optional
from torch_geometric.data import Data, Dataset
import pandas as pd

from deeprxn.data_pipeline.data_pipeline import DataPipeline


# TODO: Generalize the dataset class to non-graph datasets?
class GraphDataset(Dataset):
    """
    A flexible dataset class for molecular graphs.

    This class supports both precomputing all graphs upfront and processing
    samples on demand with optional caching.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        sample_processing_pipeline: DataPipeline,
        precompute_all: bool = True,
        cache_graphs: bool = True,
        max_cache_size: Optional[int] = None,
        subsample: Optional[int | float] = None,
    ):
        """
        Initialize the GraphDataset with the provided data and processing pipeline.

        Args:
            data (pd.DataFrame): The input data containing molecular graphs.
            sample_processing_pipeline (DataPipeline): A pipeline for processing 
            individual samples to the final representations on which the model 
            will be trained on.
            precompute_all (bool): Whether to precompute all graphs upfront.
            cache_graphs (bool): Whether to cache processed graphs (if not precomputing).
            max_cache_size (Optional[int]): Maximum size of the cache (if caching is enabled).
            subsample (Optional[int | float]): The subsample size or fraction.
        """
        self.data = self._subsample_data(data, subsample)
        self.sample_processing_pipeline = sample_processing_pipeline
        self.precompute_all = precompute_all

        if precompute_all:
            self.precomputed_graphs = [
                self._process_sample(idx) for idx in range(len(self.data))
            ]
        else:
            if cache_graphs:
                self.process_sample = lru_cache(max_cache_size)(self._process_sample)
            else:
                self.process_sample = self._process_sample

    
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
            return self.graphs[idx]
        else:
            return self.process_sample(idx)


    def _process_sample(self, idx) -> Data:
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
            return self.sample_processing_pipeline.forward(sample)
        except Exception as e:
            raise ValueError(
                f"Error processing sample {idx}, Error: {str(e)}"
            )


    def _subsample_data(self, data: pd.DataFrame, subsample: Optional[int | float]) -> pd.DataFrame:
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