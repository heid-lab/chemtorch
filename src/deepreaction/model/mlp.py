from typing import List, Optional, Callable, Union
from torch import nn
import torch

from deepreaction.act.act import Activation

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) with configurable layers and activation functions.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        hidden_dims: Optional[List[int]] = None,
        hidden_size: Optional[int] = None,
        num_layers: Optional[int] = None,
        dropout: float = 0.,
        activation: Union[nn.Module, Callable, str] = nn.ReLU,
    ):
        """
        Initialize the MLP.
        Args:
            in_channels (int): Input dimension.
            out_channels (int): Output dimension.
            hidden_dims (List[int], optional): List of hidden layer dimensions.
            hidden_size (int, optional): Hidden layer size (used if hidden_dims is not provided).
            num_layers (int, optional): Number of hidden layers (used if hidden_dims is not provided).
            dropout (float, optional): Dropout rate. Defaults to 0.
            activation (nn.Module or Callable or str, optional): Activation function. Defaults to nn.ReLU.
        """
        super().__init__()

        # Validate and resolve hidden_dims
        hidden_dims = self._resolve_hidden_dims(hidden_dims, hidden_size, num_layers)

        # Handle activation
        self.activation = self._resolve_activation(activation)
        self.dropout = dropout

        # Build layers
        self.layers = nn.Sequential()
        self.layers.append(self._construct_linear(in_channels, hidden_dims[0]))
        for i in range(1, len(hidden_dims)):
            self.layers.append(self._construct_linear(hidden_dims[i-1], hidden_dims[i]))
        self.layers.append(nn.Linear(hidden_dims[-1], out_channels))

    @staticmethod
    def _resolve_hidden_dims(hidden_dims, hidden_size, num_layers):
        if hidden_dims is not None:
            if hidden_size is not None or num_layers is not None:
                raise ValueError("Specify either hidden_dims OR hidden_size and num_layers, not both.")
            if len(hidden_dims) == 0:
                raise ValueError("hidden_dims must be a non-empty list")
            return hidden_dims
        elif hidden_size is not None and num_layers is not None:
            return [hidden_size] * num_layers
        else:
            raise ValueError("You must specify either hidden_dims OR both hidden_size and num_layers.")

    @staticmethod
    def _resolve_activation(activation):
        if isinstance(activation, nn.Module):
            return activation
        elif callable(activation):
            # Wrap callable in a nn.Module
            class LambdaActivation(nn.Module):
                def forward(self, x):
                    return activation(x)
            return LambdaActivation()
        elif isinstance(activation, str):
            return Activation(activation)
        else:
            raise ValueError("activation must be an nn.Module, a callable, or a string.")

    def _construct_linear(self, input_dim: int, output_dim: int) -> nn.Module:
        layers = []
        if self.dropout > 0:
            layers.append(nn.Dropout(self.dropout))
        layers.extend([
            nn.Linear(input_dim, output_dim),
            self.activation
        ])
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP."""
        return self.layers(x)



