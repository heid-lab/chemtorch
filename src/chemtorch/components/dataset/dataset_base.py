from functools import lru_cache
import time
from typing import Callable, Optional, Tuple, TypeVar, Generic
import pandas as pd
import torch

from chemtorch.components.representation import AbstractRepresentation
from chemtorch.components.transform import AbstractTransform
from chemtorch.components.dataset.abstact_dataset import AbstractDataset
from chemtorch.utils import enforce_base_init


# TODO: Centralize logging
# TODO: Consider saving the precomputed data objects to disk to 
# save preprocessing time for repeated runs with the same dataset.
# Note: Update precompute_time property to return 0 or time taken
# to load from disk.
T = TypeVar("T")

class DatasetBase(Generic[T], AbstractDataset[T, AbstractRepresentation[T]]):
    """
    Base class for DeepReaction datasets.

    This class implements the AbstractDataset with AbstractRepresentation[T] as the representation type.

    The dataset can handle both labeled and unlabeled data. If the input DataFrame contains a 'label' 
    column, the dataset will return tuples of (data_object, label). Otherwise, it will return only 
    the data objects.

    Warning: If the subclass inherits from multiple classes, ensure that :class:`DatasetBase` is the first 
    class in the inheritance list to ensure correct method resolution order (MRO).

    Raises:
        RuntimeError: If the subclass does not call `super().__init__()` in its `__init__()` method.
    """
    

    def __init__(
        self,
        dataframe: pd.DataFrame,
        representation: AbstractRepresentation[T],
        transform: Optional[AbstractTransform[T] | Callable[[T], T]] = None,
        precompute_all: bool = True,
        cache: bool = True,
        max_cache_size: Optional[int] = None,
        subsample: Optional[int | float] = None,
        integer_labels: bool = False,
    ):
        """
        Initialize the DatasetBase.

        Args:
            dataframe (pd.DataFrame): The input data as a pandas DataFrame. Each row represents a single sample. If the dataset 
                contains a `label` column, it will be returned alongside the computed representation. Otherwise, only the 
                representation will be returned.
            representation (AbstractRepresentation[T]): A representation instance that constructs the data object consumed by 
                the model. Must take in the fields of a single sample from the :attr:`dataframe` (row) as keyword arguments 
                and return an object of type T.
            transform (Optional[AbstractTransform[T] | Callable[[T], T]]): An optional transformation function or a composition thereof 
                (:class:`Compose`) that takes in an object of type T and returns a (possibly modified) object of 
                the same type.
            precompute_all (bool): If True, precompute all samples in the dataset. Default is True.
            cache (bool): If True, cache the processed samples. Default is True.
            max_cache_size (Optional[int]): Maximum size of the cache. Default is None.
            subsample (Optional[int | float]): The subsample size or fraction. If None, no subsampling is done.
                If an int, it specifies the number of samples to take. If a float, it specifies the fraction of
                samples to take. Default is None.
            integer_labels (bool): Whether to use integer labels (for classification) or float labels 
                (for regression). If True, labels will be torch.int64. If False, labels will be torch.float. 
                Default is False.

        Raises:
            ValueError: If the `dataframe` is not a pandas DataFrame.
            ValueError: If the `representation` is not an AbstractRepresentation instance.
            ValueError: If the `transform` is not a TransformBase, a callable, or None.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("Dataframe must be a pandas DataFrame.")
        if not isinstance(representation, AbstractRepresentation):
            raise ValueError(
                "Representation must be an instance of AbstractRepresentation."
            )
        if not isinstance(transform, (AbstractTransform, Callable, type(None))):
            raise ValueError(
                "Transform must be an instance of AbstractTransform, Callable, or None."
            )
        
        self.dataframe = self._subsample_data(dataframe, subsample)
        self.representation = representation
        self.transform = transform
        self.integer_labels = integer_labels
        self.has_labels = 'label' in self.dataframe.columns

        self.precompute_all = precompute_all
        self.precomputed_items = None
        self._precompute_time = 0.0

        if self.precompute_all:
            # print(f"INFO: Precomputing {len(self.dataframe)} items...")
            start_time = time.time()
            self.precomputed_items = [
                self._process_sample(idx) for idx in range(len(self.dataframe))
            ]
            self._precompute_time = time.time() - start_time
            # print(f"INFO: Precomputation finished in {self._precompute_time:.2f}s.")
        else:
            if cache:
                if max_cache_size is None:
                    self.process_sample = lru_cache(maxsize=None)(self._process_sample)
                else:
                    self.process_sample = lru_cache(maxsize=max_cache_size)(self._process_sample)
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

    def __getitem__(self, idx) -> T | Tuple[T, torch.Tensor]:
        """
        Retrieve a processed item by its index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            T | Tuple[T, torch.Tensor]: If the dataset has labels, returns a tuple of (data_object, label).
                Otherwise, returns only the data object of type T.
        """
        if self.precompute_all:
            if self.precomputed_items is None:
                raise RuntimeError(
                    f"Precomputed items are not available."
                )
            return self.precomputed_items[idx]
        else:
            return self.process_sample(idx)

    def get_labels(self):
        """
        Retrieve the labels for the dataset.

        Returns:
            pd.Series: The labels for the dataset if they exist.
            
        Raises:
            RuntimeError: If the dataset does not contain labels.
        """
        if not self.has_labels:
            raise RuntimeError("Dataset does not contain labels.")
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

    def _process_sample(self, idx: int) -> T | Tuple[T, torch.Tensor]:
        """
        Process a sample by its index.
        
        This method uses the representation callable to create a representation data
        object from the sample data. If a transform is provided, it applies the transform 
        to the representation object.

        Args:
            idx (int): The index of the sample to process.

        Returns:
            T | Tuple[T, torch.Tensor]: If the dataset has labels, returns a tuple of (data_object, label).
                Otherwise, returns only the data object of type T.

        Raises:
            RuntimeError: If there is an error processing the sample at the given index.
        """
        try:
            row = self.dataframe.iloc[idx]
            if self.has_labels:
                # Choose dtype based on integer_labels parameter
                label_dtype = torch.int64 if self.integer_labels else torch.float
                label = torch.tensor(row['label'], dtype=label_dtype)
                sample = row.drop("label")
            else:
                sample = row
            data_obj = self.representation(**sample)
            if self.transform:
                data_obj = self.transform(data_obj)
            
            if self.has_labels:
                return data_obj, label
            else:
                return data_obj
        except Exception as e:
            raise RuntimeError(f"Error processing sample at index {idx}: {e}")
    
    

    def __init_subclass__(cls):
        enforce_base_init(DatasetBase)(cls)
        return super().__init_subclass__()

    
    @property
    def precompute_time(self) -> float:
        """
        Get the time taken to precompute all samples.

        Returns:
            float: The time in seconds taken to precompute all samples.
        """
        if not self.precompute_all:
            raise RuntimeError("Precomputation is not enabled for this dataset.")
        return self._precompute_time

    @property
    def mean(self) -> float:
        """
        Get the mean of the labels in the dataset.

        Returns:
            float: The mean of the labels.
            
        Raises:
            RuntimeError: If the dataset does not contain labels.
        """
        if not self.has_labels:
            raise RuntimeError("Dataset does not contain labels.")
        return float(self.dataframe['label'].mean())
    
    @property
    def std(self) -> float:
        """
        Get the standard deviation of the labels in the dataset.

        Returns:
            float: The standard deviation of the labels.
            
        Raises:
            RuntimeError: If the dataset does not contain labels.
        """
        if not self.has_labels:
            raise RuntimeError("Dataset does not contain labels.")
        return float(self.dataframe['label'].std())