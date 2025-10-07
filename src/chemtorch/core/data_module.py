import builtins
from typing import Any, Callable, Dict, List, Optional, Union, TypeGuard, cast
import lightning as L
import pandas as pd
from torch.utils.data import DataLoader

try:
    from omegaconf import DictConfig
except ImportError:
    DictConfig = None

from chemtorch.core.dataset_base import DatasetBase
from chemtorch.core.property_system import DatasetProperty
from chemtorch.components.representation.abstract_representation import AbstractRepresentation
from chemtorch.components.transform.abstract_transform import AbstractTransform
from chemtorch.utils.types import AugmentationType, DataLoaderFactoryProtocol, DataSplit, DatasetKey, TransformType


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        data_pipeline: Callable[..., DataSplit | pd.DataFrame],
        representation: AbstractRepresentation,
        dataloader_factory: DataLoaderFactoryProtocol,
        transform: Optional[TransformType | Dict[DatasetKey, TransformType] | Dict[DatasetKey, Union[List[TransformType], Dict[str, TransformType]]]] = None,
        augmentations: Optional[List[AugmentationType] | Dict[str, AugmentationType]] = None,
        subsample: Optional[int | float | Dict[DatasetKey, int | float]] = None,
        precompute_all: bool = True,
        cache: bool = False,
        max_cache_size: Optional[int] = None,
        integer_labels: bool = False,
    ) -> None:
        """
        Initialize the DataModule with a data pipeline, dataset factory, and dataloader factory.

        Args:
            data_pipeline (Callable): A callable that returns a DataSplit or a pandas DataFrame.
            representation (AbstractRepresentation): An instance of a representation class to convert raw data into model-ready format.
            dataloader_factory (DataLoaderFactoryProtocol): A DataLoader class or factory function to create DataLoader instances that 
                work with the data format returned by the representation. Typically, this will be a partially initialized object that
                subclasses the torch.utils.data.DataLoader class.
            transform (Optional[Union[TransformType, Dict[DatasetKey, TransformType], Dict[DatasetKey, Union[List[TransformType], Dict[str, TransformType]]]]]):
                An optional transform or a dictionary of transforms for each stage ('train', 'val', 'test', 'predict').
                If a single transform is provided, it will be applied to all stages.
                If a dictionary is provided, it should map each dataset key to its corresponding transform.
                For the test set, a list or dict of transforms can be provided to create multiple test datasets.
                If a dict is provided, the keys should be the names of the test datasets.
            augmentations (Optional[List[AbstractAugmentation]] | Dict[str, AbstractAugmentation]):
                An optional list or dictionary of augmentations to be applied to the training dataset.
            subsample (Optional[Union[int, float, Dict[DatasetKey, Union[int, float]]]]):
                An optional integer or float to subsample the datasets. If a float is provided,
                it should be between 0 and 1 and represents the fraction of the dataset to keep.
                If an integer is provided, it represents the exact number of samples to keep.
                If a dictionary is provided, it should map each dataset key to its corresponding 
                subsample fraction or count.
            precompute_all (bool): If True, precompute all samples of the dataset. Default is True.
            cache (bool): If True, enable caching of dataset samples. Default is False.
            max_cache_size (Optional[int]): Maximum number of samples to cache. If None, cache
                size is unlimited. Default is None.
            integer_labels (bool): If True, convert labels to integers. Default is False.

        Raises:
            TypeError: If the output of the data pipeline is not a DataSplit or a pandas DataFrame.
        """
        super().__init__()

        # Initialize transforms
        def is_single_transform(t: Any) -> TypeGuard[TransformType]:
            """Type guard to check if a transform is a single transform (not list/dict)."""
            return isinstance(t, AbstractTransform) or callable(t)
        
        if transform is None:
            transform_dict = {}
        elif isinstance(transform, AbstractTransform) or callable(transform):
            transform_dict = {stage: transform for stage in ["train", "val", "test", "predict"]}
        elif isinstance(transform, dict):
            if not all(stage in ["train", "val", "test", "predict"] for stage in transform.keys()):
                raise ValueError("Transforms dictionary keys must be one of 'train', 'val', 'test', or 'predict'.")
            
            # Check individual keys - train, val, predict should be single transforms
            for key in ["train", "val", "predict"]:
                if key in transform:
                    t = transform[key]  # type: ignore  # key is guaranteed to be DatasetKey from the list above
                    if not is_single_transform(t):
                        raise TypeError(f"The value for the '{key}' key in transforms must be an AbstractTransform or callable.")
            
            # Test key can be single transform, list, or dict
            if "test" in transform:
                test_value = transform["test"]
                if isinstance(test_value, (list, dict)):
                    test_transforms = test_value.values() if isinstance(test_value, dict) else test_value
                    if not all(is_single_transform(t) for t in test_transforms):
                        raise TypeError("All elements in the list or dict of transforms for the 'test' key must be AbstractTransform instances or callables.")
                elif not is_single_transform(test_value):
                    raise TypeError("The value for the 'test' key in transforms must be an AbstractTransform, callable, a list of AbstractTransforms, or a dict of AbstractTransforms.")
            
            transform_dict = transform
        else:
            raise TypeError(f"Transforms must be either an AbstractTransform instance, callable, or a dictionary mapping dataset key to AbstractTransform instances, got {type(transform)}")

        # Validate subsample if dict and normalize subsample to a dictionary
        if isinstance(subsample, dict):
            for key in subsample.keys():
                if key not in ["train", "val", "test", "predict"]:
                    raise ValueError("Subsample dictionary keys must be one of 'train', 'val', 'test', or 'predict'.")
            subsample_dict = subsample
        else:
            subsample_dict = {k: subsample for k in ["train", "val", "test", "predict"]}

        # Initialize datasets
        def create_dataset(
            name: str,
            dataframe: pd.DataFrame,
            subsample: Optional[int | float],
            transform: Optional[Any] = None,
            augmentations: Optional[List[AugmentationType]] = None,
        ) -> DatasetBase:
            assert transform is None or is_single_transform(transform), f"{name} transform validation failed"
            return DatasetBase(
                dataframe=dataframe,
                representation=representation,  # from __init__
                name=name,
                transform=transform,
                augmentation_list=augmentations,
                subsample=subsample,
                precompute_all=precompute_all,  # from __init__
                cache=cache,                    # from __init__
                max_cache_size=max_cache_size,  # from __init__
                integer_labels=integer_labels   # from __init__
            )

        data = data_pipeline()

        if not isinstance(data, (DataSplit, pd.DataFrame)):
            raise TypeError(
                "Data pipeline must output either a DataSplit or a pandas DataFrame"
            )

        elif isinstance(data, DataSplit):
            augmentation_list = list(augmentations.values()) if isinstance(augmentations, dict) else augmentations
            self.train_dataset = create_dataset("train", data.train, subsample_dict.get("train"), transform_dict.get("train"), augmentation_list)
            self.val_dataset = create_dataset("val", data.val, subsample_dict.get("val"), transform_dict.get("val"))

            # Handle test datasets - flatten them as individual properties
            test_transforms = transform_dict.get("test")
            if isinstance(test_transforms, list):
                # Create multiple test datasets with different transforms
                self.test_dataset = create_dataset("test", data.test, subsample_dict.get("test"))  # Default without transform
                for i, transform in enumerate(test_transforms):
                    setattr(self, f"test_{i+1}_dataset", create_dataset(f"test_{i+1}", data.test, subsample_dict.get("test"), transform=transform))
            elif isinstance(test_transforms, dict):
                # Create named test datasets
                self.test_dataset = create_dataset("test", data.test, subsample_dict.get("test"))  # Default without transform
                for name, transform in test_transforms.items():
                    setattr(self, f"test_{name}_dataset", create_dataset(f"test/{name}", data.test, subsample_dict.get("test"), transform=transform))
            else:
                # Single test transform or None -> single test dataset 
                self.test_dataset = create_dataset("test", data.test, subsample_dict.get("test"), test_transforms)

        elif isinstance(data, pd.DataFrame):
            self.predict_dataset = create_dataset("predict", data, subsample_dict.get("predict"), transform_dict.get("predict"))
        
        self.dataloader_factory = dataloader_factory

    def get_dataset(self, key: str) -> DatasetBase:
        """
        Retrieve the dataset for the specified key.

        Args:
            key (str): The key for which to retrieve the dataset ('train', 'val', 'test', 'predict', 'test_<name>').

        Returns:
            DatasetBase: The dataset corresponding to the specified key.

        Raises:
            ValueError: If the dataset for the specified key is not initialized.
        """
        # NOTE: This function relies on the datasets being named 'train_dataset', 'val_dataset', etc.
        dataset = getattr(self, f"{key}_dataset", None)
        if dataset is None:
            raise ValueError(f"{key.capitalize()} dataset is not initialized.")
        return dataset

    def get_dataset_names(self) -> List[str]:
        """
        Get all available dataset names/keys.

        Returns:
            List[str]: List of all dataset keys that can be used with get_dataset() and make_dataloader().
        """
        dataset_names = []
        
        # Check for standard datasets
        for key in ["train", "val", "test", "predict"]:
            if hasattr(self, f"{key}_dataset"):
                dataset_names.append(key)
        
        # Check for additional test datasets (test_<name>)
        for attr_name in dir(self):
            if attr_name.startswith("test_") and attr_name.endswith("_dataset") and attr_name != "test_dataset":
                # Extract the dataset key (remove '_dataset' suffix)
                dataset_key = attr_name[:-8]  # Remove '_dataset'
                dataset_names.append(dataset_key)
        
        return dataset_names

    def make_dataloader(self, key: str) -> DataLoader:
        """
        Create a dataloader for the specified key.

        Args:
            key (str): The key for which to create the dataloader ('train', 'val', 'test', 'predict', 'test_<name>').

        Returns:
            DataLoader: The created dataloader for the specified dataset key.

        Raises:
            ValueError: If the dataset for the specified key is not initialized.
        """
        # if key == "test":
        #     return self._make_test_dataloader()
        # else:
        return self.dataloader_factory(
            dataset=self.get_dataset(key), 
            shuffle=(key == "train")
        )
    
    # TODO: Remove (not needed if custom trainer is implemented that passes the dataloader name directly to the routine)
    def maybe_get_test_dataloader_idx_to_suffix(self) -> Dict[int, str] | None:
        """
        If multiple named test datasets are initialized by passing a dict with named
        test set transforms, return a mapping from dataloader index to suffix.
        Otherwise, return None.

        Returns:
            Dict[int, str]: A mapping from dataloader index to suffix, or None.
            Index 0 is always the main "test" dataset (no suffix).
            Indices 1+ correspond to additional test datasets sorted alphabetically.
        """
        # Get all test dataset names in consistent order
        sorted_names = self._get_sorted_test_dataset_names()
        
        if len(sorted_names) <= 1:
            return None
        
        # Create mapping: index 0 has no suffix, others get suffixes
        idx_to_suffix = {}
        for i, name in enumerate(sorted_names):
            if i == 0:  # Main test dataset
                continue  # No suffix for index 0
            else:
                # Extract suffix from test dataset name
                if name == "test":
                    suffix = ""  # This shouldn't happen since "test" should be first
                else:
                    suffix = name.split("test_", 1)[1] if name.startswith("test_") else name
                idx_to_suffix[i] = suffix
        
        return idx_to_suffix

    ########## Lightning DataModule Methods ##############################################
    # TODO: Optionally, add `prepare_data()` to preprocess datasets and save preprocesssed data to disc.
    # TODO: Optionally, move dataset initialization to `setup()` method to allow for lazy loading.
    def train_dataloader(self):
        return self.make_dataloader("train")

    def val_dataloader(self):
        return self.make_dataloader("val")

    def test_dataloader(self):
        return self._make_test_dataloader()

    def predict_dataloader(self):
        return self.make_dataloader("predict")

    ########### Private Methods ##########################################################
    def _get_sorted_test_dataset_names(self) -> List[str]:
        """
        Get all test dataset names sorted in a consistent order.
        
        Returns:
            List[str]: Test dataset names sorted with "test" first, then others alphabetically.
        """
        # Get all test dataset names (including test_<name> variants)
        test_dataset_names = [name for name in self.get_dataset_names() if name.startswith("test")]
        
        # Sort to ensure consistent ordering: "test" first, then others alphabetically
        return sorted(test_dataset_names, key=lambda x: (x != "test", x))

    def _make_test_dataloader(self) -> DataLoader | List[DataLoader]:
        """
        Create dataloaders for the test datasets. If multiple test datasets are initialized,
        it returns a list of dataloaders, one for each test dataset.
        
        The ordering is consistent with maybe_get_test_dataloader_idx_to_suffix():
        - Index 0: main "test" dataset
        - Index 1+: additional test datasets sorted alphabetically

        Returns:
            DataLoader or List[DataLoader]: The created dataloader(s) for the test dataset(s).
        """
        # Get all test dataset names in consistent order
        sorted_names = self._get_sorted_test_dataset_names()
        
        if len(sorted_names) == 1:
            # Single test dataset
            return self.dataloader_factory(dataset=self.test_dataset, shuffle=False)
        else:
            # Multiple test datasets - create list of dataloaders in consistent order
            return [
                self.dataloader_factory(dataset=self.get_dataset(cast(DatasetKey, name)), shuffle=False)
                for name in sorted_names
            ]
