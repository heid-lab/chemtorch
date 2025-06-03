from functools import lru_cache
import time
from typing import Callable, Optional, Tuple, TypeVar, Generic
import pandas as pd
import torch
import wandb

from deepreaction.representation import AbstractRepresentation
from deepreaction.transform import AbstractTransform
from deepreaction.utils import enforce_base_init


# TODO: Centralize logging
# TODO: Consider saving the precomputed data objects to disk to 
# save preprocessing time for repeated runs with the same dataset.
# Note: Update precompute_time property to return 0 or time taken
# to load from disk.
# TODO: Generalize to unlabbeled datasets.
T = TypeVar("T")
class DatasetBase(Generic[T]):
    """
    Base class for DeepReaction datasets.

    This class defines the standard interface for datasets in the DeepReaction framework.
    All datasets should subclass :class:`DatasetBase[T]` and implement the `_get_sample_by_idx` method.    

    Warning: If the subclass inherits from multiple classes, ensure that :class:`DatasetBase` is the first 
    class in the inheritance list to ensure correct method resolution order (MRO).

    Raises:
        RuntimeError: If the subclass does not call `super().__init__()` in its `__init__()` method.

    Example:
        >>> import pandas as pd
        >>> class MyRepresentation(RepresentationBase[int]):
        ...     def __call__(self, a: int, b: int) -> int:
        ...         return a + b
        ...
        >>> class MyTransform(TransformBase[int]):
        ...     def __call__(self, data: int) -> int:
        ...         return data * 2
        ...
        >>> df = pd.DataFrame([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
        >>> dataset = DatasetBase(df, MyRepresentation, MyTransform)
        >>> result = dataset._process_sample_by_idx(0)
        >>> print(result)
        6
    """
    

    def __init__(
        self,
        dataframe: pd.DataFrame,
        representation: AbstractRepresentation[T] | Callable[..., T],
        transform: Optional[AbstractTransform[T] | Callable[[T], T]] = None,
        precompute_all: bool = True,
        cache: bool = True,
        max_cache_size: Optional[int] = None,
        subsample: Optional[int | float] = None,
    ):
        """
        Initialize the DatasetBase.

        Args:
            dataframe (pd.DataFrame): The input data as a pandas DataFrame. Each row represents a single sample.
            representation (RepresentationBase[T] | Callable[..., T]): A stateless class or callable that 
                constructs the data object consumed by the model. Must take in the fields of a single sample 
                from the :attr:`dataframe` (row) as keyword arguments and return an object of type T.
            transform (Optional[TransformBase[T] | Callable[[T], T]]): An optional transformation function or a composition thereof 
                (:class:`Compose`) that takes in an object of type T and returns a (possibly modified) object of 
                the same type.
            precompute_all (bool): If True, precompute all samples in the dataset. Default is True.
            cache (bool): If True, cache the processed samples. Default is True.
            max_cache_size (Optional[int]): Maximum size of the cache. Default is None.
            subsample (Optional[int | float]): The subsample size or fraction. If None, no subsampling is done.
                If an int, it specifies the number of samples to take. If a float, it specifies the fraction of
                samples to take. Default is None.

        Raises:
            ValueError: If the `dataframe` is not a pandas DataFrame.
            ValueError: If the `dataframe` does not contain a 'label' column.
            ValueError: If the `representation` is not a RepresentationBase or a callable.
            ValueError: If the `transform` is not a TransformBase, a callable, or None.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("dataframe must be a pandas DataFrame.")
        if "label" not in dataframe.columns:
            raise ValueError(
                f"Dataframe must contain a 'label' column, received columns: {dataframe.columns}"
            )
        if not isinstance(representation, (AbstractRepresentation, Callable)):
            raise ValueError(
                "representation must be a RepresentationBase or a callable."
            )
        if not isinstance(transform, (AbstractTransform, Callable, type(None))):
            raise ValueError(
                "transform must be a TransformBase, a callable, or None."
            )
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
                self._process_sample(idx)
                for idx in range(len(self.dataframe))
            ]
            self._precompute_time = time.time() - start_time
            print(
                f"INFO: Precomputation finished in {self._precompute_time:.2f}s."
            )
        else:
            if cache:
                self.process_sample = lru_cache(
                    max_size=max_cache_size
                )(self._process_sample)
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

    def __getitem__(self, idx) -> torch.Tensor:
        """
        Retrieve a processed item by its index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            Data: A PyTorch `Tensor` object representing the molecular fingerprint.
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
            label = torch.tensor(row['label'], dtype=torch.float)
            sample = row.drop("label")
            data_obj = self.representation(**sample)
            if self.transform:
                data_obj = self.transform(data_obj)
        except Exception as e:
            raise RuntimeError(f"Error processing sample at index {idx}: {e}")
        return data_obj, label
    

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
        """
        return self.dataframe['label'].mean().item()
    
    @property
    def std(self) -> float:
        """
        Get the standard deviation of the labels in the dataset.

        Returns:
            float: The standard deviation of the labels.
        """
        return self.dataframe['label'].std().item()