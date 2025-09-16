from asyncio import Protocol
import builtins
from typing import Any, Callable, Literal
import lightning as L
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from chemtorch.components.dataset.abstact_dataset import AbstractDataset
from chemtorch.components.dataset.dataset_base import DatasetBase
from chemtorch.utils.types import DataSplit

Stage = Literal["train", "val", "test", "predict"]


class DatasetOperationProtocol(Protocol):
    def __call__(self, dataset: AbstractDataset) -> None:
        """
        Apply an operation to the dataset and return a new dataset.
        
        Args:
            dataset: The dataset to apply the operation on.
        
        Returns:
            A new dataset after applying the operation.
        """
        pass

class DataModule(L.LightningDataModule):
    def __init__(
        self,
        data_pipeline: Callable[..., DataSplit | pd.DataFrame],
        dataset_factory: Callable[[pd.DataFrame], DatasetBase],
        dataloader_factory: Callable[[DatasetBase, bool], DataLoader],  # updated signature
    ):
        """
        Initialize the DataModule with a data pipeline, dataset factory, and dataloader factory.

        Args:
            data_pipeline (Callable): A callable that returns a DataSplit or a pandas DataFrame.
            dataset_factory (Callable): A callable that creates datasets from pandas DataFrames.
            dataloader_factory (Callable): A callable that creates DataLoader instances from datasets.
                It should accept a dataset and a boolean indicating whether to shuffle the data.

        Raises:
            TypeError: If the output of the data pipeline is not a DataSplit or a pandas DataFrame.
        """
        super().__init__()
        self.datasets = self._init_datasets(data_pipeline, dataset_factory)
        self.dataloader_factory = dataloader_factory

    def get_dataset_property(self, key: Stage, property: str) -> Any:
        """
        Retrieve a property from the dataset of the specified stage.

        Args:
            key (str): Of which dataset to get the property from:
            'train', 'val', 'test', or 'predict'.
            property (str): The name of the property to retrieve.

        Returns:
            Any: The value of the specified property from the train dataset.

        Raises:
            AttributeError: If the property does not exist in the train dataset or is not a @property.
        """
        dataset = self._get_dataset(key)
        # Check if the attribute is a property of the dataset's class
        if hasattr(type(dataset), property) and isinstance(
            getattr(type(dataset), property), builtins.property
        ):
            return getattr(dataset, property)
        else:
            raise AttributeError(
                f"Dataset does not have a property '{property}' (must be a @property)."
            )

    ########## Lightning DataModule Methods ##############################################
    # TODO: Optionally, add `prepare_data()` to preprocess datasets and save preprocesssed data to disc.
    # TODO: Optionally, move dataset initialization to `setup()` method to allow for lazy loading.
    def train_dataloader(self):
        return self._make_dataloader_or_raise("train")

    def val_dataloader(self):
        return self._make_dataloader_or_raise("val")

    def test_dataloader(self):
        return self._make_dataloader_or_raise("test")

    def predict_dataloader(self):
        return self._make_dataloader_or_raise("predict")

    ########### Private Methods ##########################################################
    def _init_datasets(
        self,
        data_pipeline: Callable[..., DataSplit | pd.DataFrame],
        dataset_factory: Callable[[pd.DataFrame, Literal["train", "val", "test", "predict"]], DatasetBase],
    ):
        """
        Initialize datasets from the data pipeline. If the data pipeline returns a DataSplit,
        it initializes train, validation, and test datasets. If it returns a pandas DataFrame,
        it initializes a predict dataset.

        Args:
            data_pipeline (Callable): A callable that returns a DataSplit or a pandas DataFrame.
            dataset_factory (Callable): A callable that creates datasets from pandas DataFrames.

        Raises:
            TypeError: If the output of the data pipeline is not a DataSplit or a pandas DataFrame.
        """
        # Note: Do not rename the dataset attributes, since _get_dataset relies on them
        # being named 'train_dataset', 'val_dataset', 'test_dataset', and 'predict_dataset'.
        data = data_pipeline()
        if isinstance(data, DataSplit):
            datasets = {
                "train": dataset_factory(data.train, "train"),
                "val": dataset_factory(data.val, "val"),
                "test": dataset_factory(data.test, "test"),
            }
        elif isinstance(data, pd.DataFrame):
            datasets = {"predict": dataset_factory(data, "predict")}
        else:
            raise TypeError(
                "Data pipeline must output either a DataSplit or a pandas DataFrame"
            )
        return datasets

    def _get_dataset(self, key: Stage) -> DatasetBase:
        """
        Retrieve the dataset for the specified key.

        Args:
            key (str): The key for which to retrieve the dataset ('train', 'val', 'test', 'predict').

        Returns:
            Any: The dataset corresponding to the specified key.

        Raises:
            ValueError: If the dataset for the specified key is not initialized.
        """
        dataset = self.datasets.get(key)
        if dataset is None:
            raise ValueError(f"{key.capitalize()} dataset is not initialized.")
        return dataset

    def _make_dataloader_or_raise(self, key: Stage) -> DataLoader:
        """
        Create a dataloader for the specified key or raise an error if the dataset is not initialized.

        Args:
            key (str): The key for which to create the dataloader ('train', 'val', 'test', 'predict').

        Returns:
            DataLoader: The created dataloader. If the key is 'train', the dataloader will shuffle the data.

        Raises:
            ValueError: If the dataset for the specified key is not initialized.
        """
        return self.dataloader_factory(
            dataset=self._get_dataset(key), 
            shuffle=(key == "train")
        )
