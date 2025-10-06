from functools import lru_cache
import logging
import math
import time
from typing import Callable, Collection, List, Literal, Optional, Tuple, TypeVar, Generic, Any
import pandas as pd
import torch

from chemtorch.components.representation import AbstractRepresentation
from chemtorch.components.transform import AbstractTransform
from chemtorch.utils import enforce_base_init
from chemtorch.utils.types import AugmentationType, DatasetKey


# TODO: Centralize logging
# TODO: Consider saving the precomputed data objects to disk to 
# save preprocessing time for repeated runs with the same dataset.
# Note: Update precompute_time property to return 0 or time taken
# to load from disk.
T = TypeVar("T")
R = TypeVar("R", bound=AbstractRepresentation)

class DatasetBase(Generic[T, R]):
    """
    Base class for ChemTorch datasets with type-safe representations.

    Type parameters:
        T: The data type produced by the representation and returned by the dataset
        R: The representation type (bounded by AbstractRepresentation)

    Type safety features:
        - The dataset returns objects of type T (or Tuple[T, torch.Tensor] with labels)
        - The representation is of type R (bounded by AbstractRepresentation)
        - Transforms must be compatible with type T
        - Static type checker can verify basic type relationships

    Usage examples:
        # Graph dataset
        DatasetBase[Data, GraphRepresentation]
        
        # Token dataset  
        DatasetBase[torch.Tensor, TokenRepresentation]
        
        # Fingerprint dataset
        DatasetBase[torch.Tensor, FingerprintRepresentation]

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
        representation: R,
        name: Optional[str] = None,
        transform: Optional[AbstractTransform[T] | Callable[[T], T]] = None,
        augmentation_list: Optional[List[AugmentationType]] = None,
        subsample: Optional[int | float] = None,
        precompute_all: bool = True,
        cache: bool = False,
        max_cache_size: Optional[int] = None,
        integer_labels: bool = False,
    ):
        """
        Initialize the DatasetBase.

        Args:
            dataframe (pd.DataFrame): The input data as a pandas DataFrame. Each row represents a single sample. If the dataset 
                contains a `label` column, it will be returned alongside the computed representation. Otherwise, only the 
                representation will be returned.
            representation (R): A representation instance that constructs the data object consumed by 
                the model. Must take in the fields of a single sample from the :attr:`dataframe` (row) as keyword arguments 
                and return an object of type T.
            name (Optional[str]): An optional name for the dataset to be used in logging (`<name> dataset`). Default is None.
            transform (Optional[AbstractTransform[T] | Callable[[T], T]]): An optional transformation function or a composition thereof 
                (:class:`Compose`) that takes in an object of type T and returns a (possibly modified) object of 
                the same type.
            augmentation_list (Optional[List[AbstractAugmentation[T] | Callable[[T, torch.Tensor], List[Tuple[T, torch.Tensor]]]]]): An optional 
                list of data augmentation functions that take in an object of type T and return a (possibly modified) object of the same type.
                Note: Augmentations are only applied to the training partition, and only if `precompute_all` is True.
            subsample (Optional[int | float]): The subsample size or fraction. If None, no subsampling is done.
                If an int, it specifies the number of samples to take. If a float, it specifies the fraction of
                samples to take. Default is None.
            precompute_all (bool): If True, precompute all samples in the dataset. Default is True.
            cache (bool): If True, cache the processed samples. Default is False.
            max_cache_size (Optional[int]): Maximum size of the cache. Default is None.
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
        if subsample is not None:
            self._validate_subsample_value(subsample)
        if not isinstance(representation, AbstractRepresentation):
            raise ValueError(
                "Representation must be an instance of AbstractRepresentation."
            )
        if not isinstance(transform, (AbstractTransform, Callable, type(None))):
            raise ValueError(
                "Transform must be an instance of AbstractTransform, Callable, or None."
            )

        self.name = name
        self.orig_df_size = len(dataframe)
        self.dataframe = self._subsample_data(dataframe, subsample)
        self.representation = representation
        self.transform = transform
        self.augmentation_list = augmentation_list
        self.integer_labels = integer_labels
        self.has_labels = 'label' in self.dataframe.columns

        self.precompute_all = precompute_all
        self.precomputed_items = None
        self.precompute_time = 0.0

        if self.precompute_all:
            start_time = time.time()
            self.precomputed_items = [
                self._apply_transform(item) for idx in range(len(self.dataframe)) 
                for item in self._make_augmentations(self._make_data_obj(idx), idx)
            ]
            self.precompute_time = time.time() - start_time
            
            # Calculate final dataset size after augmentations
            final_size = len(self.precomputed_items)
            augmentation_factor = final_size / len(self.dataframe) if len(self.dataframe) > 0 else 1
            
            # Log detailed precomputation summary
            self._log_precomputation_summary(final_size, augmentation_factor)
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

    def _validate_subsample_value(self, value: int | float):
        if not (isinstance(value, (int, float))):
            raise ValueError(f"Subsample must be an int or a float, got {type(value)}.")
        if not ((isinstance(value, int) and value > 0) or (isinstance(value, float) and 0 < value <= 1)):
            raise ValueError("Subsample values must be either a positive integer or a float between 0 and 1 (inclusive).")

    def _subsample_data(
        self, data: pd.DataFrame, 
        subsample: Optional[int | float], 
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

        if isinstance(subsample, int):
            return data.sample(n=min(subsample, len(data)))
        elif isinstance(subsample, float):
            n_samples = round(subsample * len(data))

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
            data_obj = self.representation.construct(**sample)

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
                    augmented_samples = augmentation(data_obj, label)
                    # augmented_samples is a List[tuple[T, Optional[torch.Tensor]]]
                    # We need to extend augmented_objs with these individual samples
                    augmented_objs.extend(augmented_samples)
            except Exception as e:
                raise RuntimeError(f"Error applying augmentation to sample at index {idx}: {e}")
        return augmented_objs

    def _get_dataset_name(self) -> str:
        """Get a descriptive name for the dataset for logging purposes."""
        return f"{self.name} set" if self.name else "dataset"

    def _get_transform_description(self) -> str:
        """Get a description of transforms for logging."""
        if self.transform is None:
            return "None"
        
        # Check if it's a Compose transform (list of transforms)
        if hasattr(self.transform, 'transforms') and hasattr(self.transform.transforms, '__iter__'):
            # It's a Compose transform
            transform_names = []
            for t in self.transform.transforms:
                transform_names.append(type(t).__name__)
            return " -> ".join(transform_names)
        else:
            # Single transform
            return type(self.transform).__name__

    def _get_augmentation_description(self) -> str:
        """Get a description of augmentations for logging."""
        if not self.augmentation_list or len(self.augmentation_list) == 0:
            return "None"
        
        aug_descriptions = []
        for aug in self.augmentation_list:
            if aug is not None:
                aug_name = type(aug).__name__
                # Add number of augmentations if available
                if hasattr(aug, 'num_augmentations'):
                    aug_descriptions.append(f"{aug_name}(×{aug.num_augmentations})")
                else:
                    aug_descriptions.append(aug_name)
        
        return ", ".join(aug_descriptions)

    def _log_precomputation_summary(self, final_size: int, augmentation_factor: float):
        """Log detailed precomputation summary."""
        dataset_name = self._get_dataset_name()
        
        # Main summary line
        if augmentation_factor > 1.0:
            logging.info(f"Precomputed {final_size}/{self.orig_df_size} samples for {dataset_name} in {self.precompute_time:.2f} seconds (augmentation factor: {augmentation_factor:.2f}x).")
        else:
            logging.info(f"Precomputed {final_size}/{self.orig_df_size} samples for {dataset_name} in {self.precompute_time:.2f} seconds.")
        
        # Transform details - only log if transforms exist
        if self.transform is not None:
            transform_desc = self._get_transform_description()
            logging.info(f"    Transform(s): {transform_desc}")
        
        # Augmentation details - only log if augmentations exist
        if self.augmentation_list and len(self.augmentation_list) > 0:
            aug_desc = self._get_augmentation_description()
            logging.info(f"    Augmentation(s): {aug_desc}")

    def __init_subclass__(cls):
        enforce_base_init(DatasetBase)(cls)
        return super().__init_subclass__()
