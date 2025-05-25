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