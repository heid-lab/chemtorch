from functools import lru_cache
import logging
import math
import time
from typing import Callable, Collection, List, Literal, Optional, Tuple, TypeVar, Generic
import pandas as pd
import torch

from chemtorch.components.representation import AbstractRepresentation
from chemtorch.components.transform import AbstractTransform
from chemtorch.components.augmentation import AbstractAugmentation
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
        split: Literal["train", "val", "test", "predict"],
        representation: AbstractRepresentation[T],
        transform: Optional[AbstractTransform[T] | Callable[[T], T]] = None,
        augmentation_list: List[Optional[AbstractAugmentation[T] | Callable[[T, torch.Tensor], List[Tuple[T, torch.Tensor]]]]] = None,
        precompute_all: bool = True,
        cache: bool = True,
        max_cache_size: Optional[int] = None,
        subsample: Optional[int | float] = None,
        subsample_splits: Optional[Collection[str]] = None,
        integer_labels: bool = False,
    ):
        """
        Initialize the DatasetBase.

        Args:
            dataframe (pd.DataFrame): The input data as a pandas DataFrame. Each row represents a single sample. If the dataset 
                contains a `label` column, it will be returned alongside the computed representation. Otherwise, only the 
                representation will be returned.
            split (Literal["train", "val", "test", "predict"]): The dataset split type.
            representation (AbstractRepresentation[T]): A representation instance that constructs the data object consumed by 
                the model. Must take in the fields of a single sample from the :attr:`dataframe` (row) as keyword arguments 
                and return an object of type T.
            transform (Optional[AbstractTransform[T] | Callable[[T], T]]): An optional transformation function or a composition thereof 
                (:class:`Compose`) that takes in an object of type T and returns a (possibly modified) object of 
                the same type.
            augmentation_list (Optional[List[AbstractAugmentation[T] | Callable[[T, torch.Tensor], List[Tuple[T, torch.Tensor]]]]]): An optional 
                list of data augmentation functions that take in an object of type T and return a (possibly modified) object of the same type.
                Note: Augmentations are only applied to the training partition, and only if `precompute_all` is True.
            precompute_all (bool): If True, precompute all samples in the dataset. Default is True.
            cache (bool): If True, cache the processed samples. Default is True.
            max_cache_size (Optional[int]): Maximum size of the cache. Default is None.
            subsample (Optional[int | float]): The subsample size or fraction. If None, no subsampling is done.
                If an int, it specifies the number of samples to take. If a float, it specifies the fraction of
                samples to take. Default is None.
            subsample_splits (Optional[Collection[str]]): The dataset splits to apply subsampling to. If None, subsampling is applied to all splits.
                Default is None.
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
        if split not in ["train", "val", "test", "predict"]:
            raise ValueError("split must be one of 'train', 'val', 'test', or 'predict'.")
        if subsample is not None:
            if not isinstance(subsample, (int, float)) or subsample <= 0:
                raise ValueError("subsample must be a positive int or float.")
        if not isinstance(representation, AbstractRepresentation):
            raise ValueError(
                "Representation must be an instance of AbstractRepresentation."
            )
        if not isinstance(transform, (AbstractTransform, Callable, type(None))):
            raise ValueError(
                "Transform must be an instance of AbstractTransform, Callable, or None."
            )
        
        self.dataframe = self._subsample_data(dataframe, split, subsample, subsample_splits)
        self.representation = representation
        self.transform = transform
        self.augmentation_list = augmentation_list if split == "train" else None
        self.integer_labels = integer_labels
        self.has_labels = 'label' in self.dataframe.columns

        self.precompute_all = precompute_all
        self.precomputed_items = None
        self._precompute_time = 0.0

        if self.precompute_all:
            # print(f"INFO: Precomputing {len(self.dataframe)} items...")
            start_time = time.time()
            self.precomputed_items = [
                self._apply_transform(item) for idx in range(len(self.dataframe)) 
                for item in self._make_augmentations(self._make_data_obj(idx), idx)
            ]
            self._precompute_time = time.time() - start_time
            # print(f"INFO: Precomputation finished in {self._precompute_time:.2f}s.")
        else:
            process_sample = lambda idx: self._apply_transform(self._make_data_obj(idx))
            if cache:

                self.process_sample = lru_cache(maxsize=max_cache_size)(process_sample)
            else:
                self.process_sample = process_sample

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
        self, data: pd.DataFrame, 
        split: Literal["train", "val", "test", "predict"], 
        subsample: Optional[int | float], 
        subsample_splits: Optional[Collection[str]]
    ) -> pd.DataFrame:
        """
        Subsample the data.

        Args:
            data (pd.DataFrame): The original data.
            split (Literal["train", "val", "test", "predict"]): The dataset split type.
            subsample (Optional[int | float]): The subsample size or fraction.
            subsample_splits (Optional[Collection[str]]): The dataset splits to apply subsampling to. If None, subsampling is applied to all splits.
        Returns:
            pd.DataFrame: The subsampled data.
        """
        if subsample is None or subsample == 1.0 or (subsample_splits is not None and split not in subsample_splits):
            return data

        if isinstance(subsample, int):
            return data.sample(n=min(subsample, len(data)))
        elif isinstance(subsample, float):
            n_samples = round(subsample * len(data))
            logging.info(f"Subsampling {split} set: fraction={subsample}, original size={len(data)}, subsampled size={n_samples}.")
            
            # Ensure at least 1 sample if the original data is not empty and subsample > 0
            if n_samples == 0 and len(data) > 0 and subsample > 0:
                logging.warning(f"Subsample fraction {subsample} too small for dataset size {len(data)}, rounding up to 1 sample.")
                n_samples = 1
            
            return data.sample(n=min(n_samples, len(data)))
        else:
            raise ValueError("Subsample must be an int or a float.")

    def _make_data_obj(self, idx: int) -> T | Tuple[T, torch.Tensor]:
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

            if self.has_labels:
                return data_obj, label
            else:
                return data_obj
        except Exception as e:
            raise RuntimeError(f"Error processing sample at index {idx}: {e}")

    def _apply_transform(self, sample: T | Tuple[T, torch.Tensor]) -> T | Tuple[T, torch.Tensor]:
        """
        Apply the transform to the sample if a transform is provided.

        Args:
            sample (T | Tuple[T, torch.Tensor]): The sample to transform.
        
        Returns:
            T | Tuple[T, torch.Tensor]: The transformed sample.
        """
        if self.transform is None:
            return sample

        if isinstance(sample, tuple):
            data_obj, label = sample
            transformed_data = self.transform(data_obj)
            return transformed_data, label
        else:
            return self.transform(sample)

    def _make_augmentations(self, sample: T | Tuple[T, torch.Tensor], idx: int) -> List[T | Tuple[T, torch.Tensor]]:
        """
        Apply augmentation to the sample if augmentation is provided.
        
        Args:
            sample (T | Tuple[T, torch.Tensor]): The sample to augment.
                
        Returns:
            List[T | Tuple[T, torch.Tensor]]: A list of samples containing at least the original sample,
                and possibly augmented samples if any augmentations are applied.
                If the dataset has labels, each item in the list is a tuple of (data_object, label).
                Otherwise, each item in the list is only the data object of type T.

        Raises:
            RuntimeError: If there is an error processing the sample at the given index.
        """
        augmented_objs = [sample]

        if self.augmentation_list and len(self.augmentation_list) > 0:
            try:
                label = None
                if isinstance(sample, tuple):
                    data_obj, label = sample
                else:
                    data_obj = sample
                for augmentation in self.augmentation_list:
                    if augmentation is None:
                        continue
                    augmented_sample = augmentation(data_obj, label)
                    augmented_objs.append(augmented_sample)
            except Exception as e:
                raise RuntimeError(f"Error applying augmentation to sample at index {idx}: {e}")
        return augmented_objs

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