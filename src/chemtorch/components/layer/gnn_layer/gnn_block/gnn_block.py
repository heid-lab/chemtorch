from typing import Any, Callable, Dict, Optional, Union

from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn.resolver import activation_resolver

from chemtorch.components.layer.utils import (
    ResidualConnection,
    init_dropout,
    init_norm,
    normalize,
)
from chemtorch.utils.decorators.enforce_base_init import enforce_base_init


class GNNBlock(nn.Module):
    """
    Base class for GNN blocks.

    This class provides utility functions and a template for implementing
    a GNN blocks.
    Subclasses should call `super().__init__()` in their `__init__` method to
    pass initialization args to `GNNBlock`, and override the `forward`
    method to implement custom lo

    Citation:
    Luo, Y., Shi, L., & Wu, X. M. (2025). Unlocking the Potential of Classic
    GNNs for Graph-level Tasks: Simple Architectures Meet Excellence.
    https://arxiv.org/abs/2502.09263

    Raises:
        RuntimeError: If the subclass does not call `super().__init__()` in its
            `__init__` method.
    """

    def __init__(
        self,
        graph_conv: nn.Module,
        residual: bool = False,
        ffn: bool = False,
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        hidden_channels: int = None,
    ):
        """
        Initializes the GNNBlock.

        Args:
            graph_conv (nn.Module): The graph convolution.
            residual (bool): Whether to use a residual connection for the graph convolution (default: `False`).
            dropout (float): Dropout rate for the FFN (default: `0.0`).
            ffn (bool): Whether to add two linear layers after the graph convolion (default: `False`).
            act (str or Callable, optional): The non-linear activation function to
                use. (default: :obj:`"relu"`)
            act_kwargs (Dict[str, Any], optional): Arguments passed to the
                respective activation function defined by :obj:`act`.
                (default: :obj:`None`)
            norm (str or Callable, optional): The normalization function to
                use, as implemted in PyTorch Geometric (default: `None`)
            norm_kwargs (Dict[str, Any], optional): Arguments passed to the
                respective normalization function defined by :obj:`norm`.
                (default: :obj:`None`)

        Raises:
            ValueError: If `hidden_channels` is not specified when using normalization or FFN.
            ValueError: If `fnn_dropout_rate` is less than `0`.
        """
        super(GNNBlock, self).__init__()

        if norm and hidden_channels is None:
            raise ValueError(
                "hidden_channels must be specified when using normalization."
            )
        if ffn and hidden_channels is None:
            raise ValueError("hidden_channels must be specified when using FFN.")

        self.graph_conv = graph_conv
        self.activation = activation_resolver(act, **(act_kwargs or {}))

        # Opt-ins
        self.norm = init_norm(norm, hidden_channels, norm_kwargs)
        self.residual = ResidualConnection(residual)
        self.dropout = init_dropout(dropout)

        self.use_ffn = ffn
        if self.use_ffn:
            self.ffn_norm_in = init_norm(norm, hidden_channels, norm_kwargs)
            self.ffn_linear1 = nn.Linear(hidden_channels, hidden_channels * 2)
            self.ffn_linear2 = nn.Linear(hidden_channels * 2, hidden_channels)
            self.ffn_act_fn = activation_resolver(act, **(act_kwargs or {}))
            self.ffn_norm_out = init_norm(norm, hidden_channels, norm_kwargs)
            self.ffn_residual = ResidualConnection(use_residual=True)
            self.ffn_dropout1 = init_dropout(dropout)
            self.ffn_dropout2 = init_dropout(dropout)

    def forward(self, batch: Batch) -> Batch:
        """
        Forward pass of the GNNBlockLayer.

        Args:
            batch (Batch): The input batch of data.

        Returns:
            Batch: The output batch after processing.
        """
        # Register input for residual connection
        self.residual.register(batch.x)

        # Message passing
        batch = self.graph_conv(batch)

        # Optional normalization
        batch.x = normalize(batch.x, batch, self.norm)
        # Activation
        batch.x = self.activation(batch.x)

        # Optional dropout
        if self.dropout is not None:
            batch.x = self.dropout(batch.x)

        # Optional residual connection
        batch.x = self.residual.apply(batch.x)

        # Optional Feed-Forward Network (FFN)
        if self.use_ffn:
            self.ffn_residual.register(batch.x)
            batch.x = normalize(batch.x, batch, self.ffn_norm_in)
            batch.x = self.ffn_dropout1(self.ffn_act_fn(self.ffn_linear1(batch.x)))
            batch.x = self.ffn_dropout2(self.ffn_linear2(batch.x))
            batch.x = self.ffn_residual.apply(batch.x)
            batch.x = normalize(batch.x, batch, self.ffn_norm_out)

        return batch

    def __init_subclass__(cls) -> None:
        enforce_base_init(GNNBlock)(cls)
        return super().__init_subclass__()
