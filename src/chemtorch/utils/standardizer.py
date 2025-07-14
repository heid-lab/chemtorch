from typing import Union
import numpy as np
import torch


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

    def standardize(self, x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        Standardize the input data by subtracting the mean and dividing by the standard deviation.

        Args:
            x (Union[torch.Tensor, np.ndarray]): Input data to standardize.

        Returns:
            Union[torch.Tensor, np.ndarray]: Standardized data.
        """
        return (x - self.mean) / self.std
    
    def destandardize(self, x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        Reverse the standardization of the input data by multiplying by the standard deviation

        Args:
            x (Union[torch.Tensor, np.ndarray]): Input data to reverse standardize.

        Returns:
            Union[torch.Tensor, np.ndarray]: Reverse standardized data.
        """
        return (x * self.std) + self.mean

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