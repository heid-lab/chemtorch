from abc import ABC, abstractmethod
from typing import Any, List, NamedTuple, Union

import numpy as np
import pandas as pd
import torch

class DataSplit(NamedTuple):
    """
    A named tuple to hold the data splits for training, validation, and testing.
    """
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


class DataPipelineModule(ABC):
    """
    """
    def forward(self):
        """
        The forward method is called to execute the module's functionality.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Calls the forward method when the module is called.
        """
        return self.forward(*args, **kwargs)

class DataReader(DataPipelineModule):
    """
    Abstract base class for data readers.
    This class defines the interface for reading data from various sources.
    """

    @abstractmethod
    def forward(self):
        raise NotImplementedError("Subclasses should implement this method.")


class DataSplitter(DataPipelineModule):
    """
    Abstract base class for data splitting strategies.
    """

    @abstractmethod
    def forward(self, raw) -> DataSplit:
        """
        Splits the raw data into training, validation, and test partitions.

        Args:
            raw: The raw data to be split.

        Returns:
            DataSplit: A named tuple containing the train, val, and test dataframes.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    
class DataPipeline(DataPipelineModule):
    """
    A class to manage the data pipeline, which consists of multiple modules.
    """
    def __init__(self, modules: List[DataPipelineModule]):
        """
        Initializes the DataPipeline with a list of modules.

        The modules are executed in the order they are provided in the list.
        The first module is expected to return the raw data, and the subsequent modules will process this data.
        The last module is expected to return a DataSplit object.

        Args:
            modules (List[DataPipelineModule]): A list of modules to be executed in the pipeline.
        """
        self.modules = modules 

    def forward(self) -> DataSplit:
        """
        Executes the data pipeline by calling the forward method of each module in order.

        Returns:
            DataSplit: A named tuple containing the train, val, and test dataframes.
        """
        data = self.modules[0].forward()
        for module in self.modules[1:]:
            data = module.forward(data)

        assert isinstance(data, DataSplit), "Final output must be a DataSplit object"
        return data

class Standardizer:
    def __init__(self, mean: Union[float, torch.Tensor, np.ndarray], std: Union[float, torch.Tensor, np.ndarray]) -> None:
        """
        Create a standardizer to standardize sample using the given mean and standard deviation.

        Args:
            mean (Union[float, torch.Tensor, np.ndarray]): Mean value(s) for standardization.
            std (Union[float, torch.Tensor, np.ndarray]): Standard deviation value(s) for standardization.

        Raises:
            TypeError: If mean or std are not of type torch.Tensor or np.ndarray, or if they are not of the same type.
            ValueError: If mean and std do not have the same shape.
        """
        self.validate(mean, std)
        self.mean = mean
        self.std = std

    def __call__(self, x: Union[torch.Tensor, np.ndarray], rev=False) -> Union[torch.Tensor, np.ndarray]:
        """
        Standardize or reverse standardize the input data.

        Args:
            x (Union[torch.Tensor, np.ndarray]): Input data to standardize.
            rev (bool): If True, reverse the standardization.

        Returns:
            Union[torch.Tensor, np.ndarray]: Standardized or reverse standardized data.

        Raises:
            TypeError: If input data is not of the same type as mean and std.
            ValueError: If input data does not have the same shape as mean and std.
        """
        if not isinstance(self.mean, float):
            if not isinstance(x, type(self.mean)):
                raise TypeError("x must be of the same type as mean and std, unless mean and std are floats.")
            if x.shape != self.mean.shape:
                raise ValueError("x must have the same shape as mean and std, unless mean and std are floats.")
        if rev:
            return (x * self.std) + self.mean
        return (x - self.mean) / self.std

    @staticmethod
    def validate(
        mean: Union[float, torch.Tensor, np.ndarray],
        std: Union[float, torch.Tensor, np.ndarray],
    ) -> None:
        """
        Validate the mean and standard deviation values.

        Args:
            mean (Union[float, torch.Tensor, np.ndarray]): Mean value(s) for standardization.
            std (Union[float, torch.Tensor, np.ndarray]): Standard deviation value(s) for standardization.

        Raises:
            TypeError: If mean or std are not of type torch.Tensor or np.ndarray, or if they are not of the same type.
            ValueError: If mean and std do not have the same shape.
        """
        if not isinstance(mean, (float, torch.Tensor, np.ndarray)):
            raise TypeError("Mean must be a torch.Tensor or np.ndarray.")
        if not isinstance(std, (float, torch.Tensor, np.ndarray)):
            raise TypeError("Standard deviation must be a float, torch.Tensor or np.ndarray.")
        if type(mean) != type(std):
            raise TypeError("Mean and standard deviation must be of the same type.")
        if not isinstance(mean, float) and mean.shape != std.shape:
            raise ValueError("Mean and standard deviation must have the same shape.")
