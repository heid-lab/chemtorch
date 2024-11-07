from typing import Literal

import torch
import torch.nn as nn

ActivationType = Literal["relu",]


class Activation(nn.Module):
    """Base class for activation functions with Hydra support."""

    def __init__(
        self,
        activation_type: ActivationType = "relu",
        inplace: bool = False,
        **kwargs,
    ):
        """Initialize activation function.

        Args:
            activation_type: Type of activation function
            inplace: Whether to perform the operation in-place
            **kwargs: Additional arguments for specific activations
        """
        super().__init__()
        self.activation_type = activation_type
        self.inplace = inplace

        if activation_type == "relu":
            self.act = nn.ReLU(inplace=inplace)

        else:
            raise ValueError(f"Unknown activation type: {activation_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)
