from typing import Any, Callable, Dict, List, Optional, Union
from torch import nn
import torch
from torch_geometric.nn.resolver import activation_resolver

from deepreaction.model.abstract_model import DeepReactionModel

# TODO: Remove and use `torch_geometric.nn.MLP` instead
class MLP(nn.Module, DeepReactionModel[torch.Tensor]):
    """
    Multi-Layer Perceptron (MLP) with configurable layers and activation functions.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        hidden_dims: Optional[List[int]] = None,
        hidden_size: Optional[int] = None,
        num_hidden_layers: Optional[int] = None,
        dropout: float = 0.,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialized an MLP.

        The MLP will have the following structure:
        - Input layer: `in_channels` -> `hidden_dims[0]`
        - Hidden layers: `hidden_dims[i-1]` -> `hidden_dims[i]` for each `i` in `range(1, len(hidden_dims))`
        - Output layer: `hidden_dims[-1]` -> `out_channels`

        The activation function is applied after each linear layer, except the last one.

        The number and dimensions of hidden layers can be specified in two ways:
        1. By providing a list of integers `hidden_dims`, where each integer specifies the number of neurons in that layer.
        2. By providing a single integer `hidden_size` and the number of hidden layers `num_hidden_layers`, which will create
           a uniform MLP with `num_hidden_layers` hidden layers (`hidden_size` -> `hidden_size`).

        Args:
            in_channels (int): Input dimension.
            out_channels (int): Output dimension.
            hidden_dims (List[int], optional): List of hidden layer dimensions (preferred).
            hidden_size (int, optional): Hidden layer size (used if `hidden_dims` is not provided).
            num_hidden_layers (int, optional): Number of hidden layers (used if `hidden_dims` is not provided).
            dropout
              (float, optional): Dropout rate. Defaults to 0.
            act (str or Callable, optional): Activation function. Defaults to "relu".
            act_kwargs (Dict[str, Any], optional): Arguments for the activation function. Defaults to None.

        Raises:
            ValueError: If both `hidden_dims` and `hidden_size` or `num_hidden_layers` are provided.
            ValueError: If `hidden_dims` is not a list or is empty.
            ValueError: If `num_hidden_layers` is less than 1.

        Example:
            >>> # 2 layer MLP
            >>> mlp = MLP(
            ...     in_channels=128,
            ...     out_channels=10,
            ...     hidden_dims=[64],
            ...     dropout_rate=0.5,
            ... )
            >>> print(mlp)
            MLP(
              (layers): Sequential(
                (0): Sequential(
                  (0): Dropout(p=0.5, inplace=False)
                  (1): Linear(in_features=128, out_features=64, bias=True)
                  (2): ReLU()
                )
                (1): Sequential(
                  (0): Dropout(p=0.5, inplace=False)
                  (1): Linear(in_features=64, out_features=10, bias=True)
                )
              )
            )
        """
        super().__init__()

        # Validate and resolve hidden_dims
        hidden_dims = self._resolve_hidden_dims(hidden_dims, hidden_size, num_hidden_layers)

        self.activation = activation_resolver(act, **(act_kwargs or {}))
        self.dropout = dropout

        # Build layers
        self.layers = nn.Sequential()
        if not hidden_dims:  # Single-layer perceptron
            self.layers.append(nn.Linear(in_channels, out_channels))
        else:
            self.layers.append(self._construct_linear(in_channels, hidden_dims[0]))
            for i in range(1, len(hidden_dims)):
                self.layers.append(self._construct_linear(hidden_dims[i-1], hidden_dims[i]))
            self.layers.append(nn.Linear(hidden_dims[-1], out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP."""
        return self.layers(x)

    @staticmethod
    def _resolve_hidden_dims(hidden_dims, hidden_size, num_hidden_layers):
        # Predefined error messages
        val_err_misspecified_args = ValueError("Specify either hidden_dims OR hidden_size and num_hidden_layers, not both.")

        if hidden_dims is None and hidden_size is None and num_hidden_layers is None:
            return []

        if hidden_dims is not None:
            if hidden_size is not None or num_hidden_layers is not None:
                raise val_err_misspecified_args
            if not isinstance(hidden_dims, list):
                raise ValueError("hidden_dims must be a list or None")
            return hidden_dims
        else:
            if num_hidden_layers is not None:
                if num_hidden_layers < 1:
                    return []
                elif hidden_size is None:
                    raise val_err_misspecified_args
                return [hidden_size] * num_hidden_layers

    def _construct_linear(self, input_dim: int, output_dim: int) -> nn.Module:
        layers = []
        if self.dropout > 0:
            layers.append(nn.Dropout(self.dropout))
        layers.append(nn.Linear(input_dim, output_dim))
        if self.activation is not None:
            layers.append(self.activation)
        return nn.Sequential(*layers)




