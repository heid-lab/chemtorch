from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch import nn
from torch_geometric.nn.resolver import activation_resolver

from chemtorch.components.layer.utils import init_norm


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
        num_hidden_layers: Optional[int] = None,
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes an MLP. The architecture is built to match FFNHead's structure
        and instantiation order precisely.

        - A Dropout layer is placed before every Linear layer.
        - Activation is applied after every hidden Linear layer.
        - Normalization is applied after every hidden Linear layer (if specified).

        Args:
            in_channels (int): Input dimension.
            out_channels (int): Output dimension.
            hidden_dims (List[int], optional): List of hidden layer dimensions (preferred).
            hidden_size (int, optional): Hidden layer size (used if `hidden_dims` is not provided).
            num_hidden_layers (int, optional): Number of hidden layers (used if `hidden_dims` is not provided).
            dropout (float, optional): Dropout rate. Defaults to 0.
            act (str or Callable, optional): Activation function. Defaults to "relu".
            act_kwargs (Dict[str, Any], optional): Arguments for the activation function.
            norm (str or Callable, optional): Normalization layer. Defaults to None.
            norm_kwargs (Dict[str, Any], optional): Arguments for the normalization layer.
        """
        super().__init__()

        # Validate and resolve hidden_dims
        hidden_dims = self._resolve_hidden_dims(
            hidden_dims, hidden_size, num_hidden_layers
        )

        self.activation = activation_resolver(act, **(act_kwargs or {}))
        self.norm = norm
        self.norm_kwargs = norm_kwargs or {}
        self.dropout = dropout

        layers = []
        current_dim = in_channels

        # [Dropout, Linear, Activation, Normalization]
        for hidden_dim in hidden_dims:
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(current_dim, hidden_dim))
            if self.activation is not None:
                layers.append(self.activation)
            if self.norm is not None:
                layers.append(
                    init_norm(self.norm, hidden_dim, **self.norm_kwargs)
                )
            current_dim = hidden_dim

        # [Dropout, Linear]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(current_dim, out_channels))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP."""
        return self.layers(x)

    @staticmethod
    def _resolve_hidden_dims(
        hidden_dims: Optional[List[int]], 
        hidden_size: Optional[int], 
        num_hidden_layers: Optional[int]
    ) -> List[int]:
        # Predefined error messages
        val_err_misspecified_args = ValueError(
            "Specify either hidden_dims OR hidden_size and num_hidden_layers, not both."
        )

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
            else:
                raise val_err_misspecified_args
