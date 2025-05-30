import builtins
from typing import Any, Callable, Literal
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
        """Download and preprocess dataset here if needed."""
        pass

    def setup(self, stage: str = None):
        dp_out = self.data_pipeline()
        self._assign_datasets_from_pipeline_output(dp_out, stage)

    def train_dataloader(self):
        return self._make_dataloader('train')
        
    def val_dataloader(self):
        return self._make_dataloader('val')
        
    def test_dataloader(self):
        return self._make_dataloader('test')
    
    def predict_dataloader(self):
        return self._make_dataloader('predict')
    
    def get_dataset_property(
            self, 
            stage: Literal['train', 'val', 'test', 'predict'],
            property: str
        ) -> Any:
        """
        Retrieve a property from the dataset of the specified stage.

        Args:
            stage (str): Of which dataset to get the property from:
            'train', 'val', 'test', or 'predict'.
            property (str): The name of the property to retrieve.

        Returns:
            Any: The value of the specified property from the train dataset.

        Raises:
            AttributeError: If the property does not exist in the train dataset or is not a @property.
        """
        dataset = self._get_dataset(stage)
        # Check if the attribute is a property of the dataset's class
        if hasattr(type(dataset), property) and isinstance(getattr(type(dataset), builtins.property), property):
            return getattr(dataset, property)
        else:
            raise AttributeError(f"Dataset does not have a property '{property}' (must be a @property).")

    def _assign_datasets_from_pipeline_output(self, dp_out, stage: str = None):
        """
        Assign datasets based on the type of dp_out and the stage.
        """
        if stage is not None:
            if stage in ['train', 'val', 'test']:
                if not isinstance(dp_out, DataSplit):
                    raise ValueError(
                        "Data pipeline output must be DataSplit for 'train', 'val', or 'test' stages."
                    )
                self.train_dataset = self.dataset_factory(dp_out.train)
                self.val_dataset = self.dataset_factory(dp_out.val)
                self.test_dataset = self.dataset_factory(dp_out.test)
            elif stage == 'predict':
                if not isinstance(dp_out, pd.DataFrame):
                    raise ValueError(
                        "Data pipeline output must be a pandas DataFrame for 'predict' stage."
                    )
                self.predict_dataset = self.dataset_factory(dp_out)
            else:
                raise ValueError(f"Unknown stage: {stage}")
        else:
            # No stage specified: assign based on type only
            if isinstance(dp_out, DataSplit):
                self.train_dataset = self.dataset_factory(dp_out.train)
                self.val_dataset = self.dataset_factory(dp_out.val)
                self.test_dataset = self.dataset_factory(dp_out.test)
            elif isinstance(dp_out, pd.DataFrame):
                self.predict_dataset = self.dataset_factory(dp_out)
            else:
                raise TypeError(
                    "Data pipeline output must be a DataSplit or a pandas DataFrame"
                )
        
    def _get_dataset(self, stage: str):
        if not stage in ['train', 'val', 'test', 'predict']:
            raise ValueError(f"Unknown stage: {stage}. Must be one of 'train', 'val', 'test', or 'predict'.")
        dataset_attr = f"{stage}_dataset"
        if not hasattr(self, dataset_attr):
            raise RuntimeError(f"{stage.capitalize()} dataset is not set up. Ensure data pipeline outputs expected type and setup() is called.")
        return getattr(self, dataset_attr)
    
    def _make_dataloader(self, stage: str):
        return self.dataloader_factory(
            dataset=self._get_dataset(stage),
            shuffle=(stage == 'train'),
        )
