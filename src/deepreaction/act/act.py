from typing import Literal

import torch
import torch.nn as nn

ActivationType = Literal["relu", "leaky_relu", "identity"]


class Activation(nn.Module):

    def __init__(
        self,
        activation_type: ActivationType = "relu",
        inplace: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.activation_type = activation_type
        self.inplace = inplace

        if activation_type == "relu":
            self.act = nn.ReLU(inplace=inplace)
        elif activation_type == "leaky_relu":
            self.act = nn.LeakyReLU(inplace=inplace)
        elif activation_type == "identity":
            self.act = nn.Identity()

        else:
            raise ValueError(f"Unknown activation type: {activation_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)
