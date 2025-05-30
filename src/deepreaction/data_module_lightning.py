from typing import Any, Callable
import lightning as L
import pandas as pd

from deepreaction.utils.data_split import DataSplit

class DataModule(L.LightningDataModule):
    def __init__(
        self,
        data_pipeline: Callable[..., DataSplit],
        dataset_factory: Callable[[pd.DataFrame], Any],
        dataloader_factory: Callable,
    ):
        self.data_pipeline = data_pipeline
        self.dataset_factory = dataset_factory
        self.dataloader_factory = dataloader_factory

    def prepare_data(self):
        """
        TODO: Download and preprocess dataset here if needed.
        """

    def setup(self, stage: str = None):
        # TODO: Load standardizer or compute mean and std from training data
        dp_out = self.data_pipeline()
        if isinstance(dp_out, DataSplit):
            dataframes = dp_out
            self.train_dataset = self.dataset_factory(dataframes.train)
            self.val_dataset = self.dataset_factory(dataframes.val)
            self.test_dataset = self.dataset_factory(dataframes.test)
        elif isinstance(dp_out, pd.DataFrame):
            self.predict_dataset = self.dataset_factory(dp_out)
        else:
            raise TypeError(
                "Data pipeline output must be a DataSplit or a pandas DataFrame"
            )

    def train_dataloader(self):
        return self._make_dataloader('train')
        
    def val_dataloader(self):
        return self._make_dataloader('val')
        
    def test_dataloader(self):
        return self._make_dataloader('test')
    
    def predict_dataloader(self):
        return self._make_dataloader('predict')
    
    def _get_dataset(self, stage: str):
        if not stage in ['train', 'val', 'test', 'predict']:
            raise ValueError(f"Unknown stage: {stage}. Must be one of 'train', 'val', 'test', or 'predict'.")
        dataset_attr = f"{stage}_dataset"
        if not hasattr(self, dataset_attr):
            raise RuntimeError(f"{stage.capitalize()} dataset is not set up. Call setup() first.")
        return getattr(self, dataset_attr)
    
    def _make_dataloader(self, stage: str):
        return self.dataloader_factory(
            dataset=self._get_dataset(stage),
            shuffle=(stage == 'train'),
        )

    def get_dataset_property(self, stage: str, property: str) -> Any:
        """
        Retrieve a property from the dataset of a given stage.

        Args:
            stage (str): Of which dataset to get the property from:
            'train', 'val', 'test', or 'predict'.
            property (str): The name of the property to retrieve.

        Returns:
            Any: The value of the specified property from the train dataset.

        Raises:
            AttributeError: If the property does not exist in the train dataset.
        """
        dataset = self._get_dataset(stage)
        if hasattr(dataset, property):
            return getattr(dataset, property)
        else:
            raise AttributeError(f"Dataset does not have attribute '{property}'.")