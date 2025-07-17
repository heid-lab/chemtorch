import builtins
from typing import Any, Callable, Literal
import lightning as L
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from chemtorch.utils.data_split import DataSplit


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        data_pipeline: Callable[..., DataSplit | pd.DataFrame],
        dataset_factory: Callable[[pd.DataFrame], Dataset],
        dataloader_factory: Callable[[Dataset, bool], DataLoader],
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
        self._init_datasets(data_pipeline, dataset_factory)
        self.dataloader_factory = dataloader_factory

    def get_dataset_property(
        self, key: Literal["train", "val", "test", "predict"], property: str
    ) -> Any:
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
        dataset_factory: Callable[[pd.DataFrame], Dataset],
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
            self.train_dataset = dataset_factory(data.train)
            self.val_dataset = dataset_factory(data.val)
            self.test_dataset = dataset_factory(data.test)
        elif isinstance(data, pd.DataFrame):
            self.predict_dataset = dataset_factory(data)
        else:
            raise TypeError(
                "Data pipeline must output either a DataSplit or a pandas DataFrame"
            )

    def _get_dataset(self, key: Literal["train", "val", "test", "predict"]):
        """
        Retrieve the dataset for the specified key.

        Args:
            key (str): The key for which to retrieve the dataset ('train', 'val', 'test', 'predict').

        Returns:
            Any: The dataset corresponding to the specified key.

        Raises:
            ValueError: If the dataset for the specified key is not initialized.
        """
        dataset = getattr(self, f"{key}_dataset", None)
        if dataset is None:
            raise ValueError(f"{key.capitalize()} dataset is not initialized.")
        return dataset

    def _make_dataloader_or_raise(
        self,
        key: Literal["train", "val", "test", "predict"],
    ):
        """
        Create a dataloader for the specified key or raise an error if the dataset is not initialized.

        Args:
            key (str): The key for which to create the dataloader ('train', 'val', 'test', 'predict').

        Returns:
            DataLoader: The created dataloader.

        Raises:
            ValueError: If the dataset for the specified key is not initialized.
        """
        return self.dataloader_factory(
            dataset=self._get_dataset(key), shuffle=(key == "train")
        )
