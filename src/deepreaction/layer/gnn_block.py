from typing import Any, Callable, Dict, Union
from torch import nn
from typing import Optional

from torch_geometric.data import Batch
from torch_geometric.nn.resolver import activation_resolver

from deepreaction.layer.utils import ResidualConnection, init_2_layer_ffn, init_dropout, init_norm, normalize
from deepreaction.utils import enforce_base_init

class GNNBlockLayer(nn.Module):
    """
    Base class for GNN block layers.

    This class provides utility functions and a template for implementing 
    a GNN block layer.
    Subclasses should call `super().__init__()` in their `__init__` method to 
    pass initialization args to `GNNBlockLayer`, and override the `forward` 
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
        mpnn: nn.Module,
        use_mpnn_residual: bool = False,
        use_ffn: bool = False,
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        hidden_channels: int = None,
        **kwargs: Any,  # TODO: Add suppoert for PNA 'degree_statistics'
    ):
        """
        Initializes the GNNBlockLayer.

        Args:
            mpnn (nn.Module): The message passing neural network (MPNN) layer.
            use_mpnn_residual (bool): Whether to use a residual connection for the MPNN (default: `False`).
            use_ffn (bool): Whether to use a feed-forward network after the MPNN layer (default: `False`).
            dropout (float): Dropout rate for the FFN (default: `0.0`).
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
            hidden_channels (int): Number of hidden channels (needs to be specified 
                if a norm or FFN is use. Default: `None`).

        Raises:
            ValueError: If `hidden_channels` is not specified when using normalization or FFN.
            ValueError: If `fnn_dropout_rate` is less than `0`.
        """
        super(GNNBlockLayer, self).__init__()

        if norm and hidden_channels is None:
            raise ValueError("hidden_channels must be specified when using normalization.")
        if use_ffn and hidden_channels is None:
            raise ValueError("hidden_channels must be specified when using FFN.")

        self.mpnn = mpnn
        self.activation = activation_resolver(act, **(act_kwargs or {}))

        # Opt-ins
        self.mpnn_norm = init_norm(norm, hidden_channels, norm_kwargs)
        self.mpnn_residual = ResidualConnection(use_mpnn_residual)
        self.dropout = init_dropout(dropout)

        self.use_ffn = use_ffn
        if self.use_ffn:
            self.ffn = init_2_layer_ffn(hidden_channels, dropout, self.activation)
            self.ffn_norm_in = init_norm(norm, hidden_channels, norm_kwargs)
            self.ffn_norm_out = init_norm(norm, hidden_channels, norm_kwargs)
            self.ffn_residual = ResidualConnection(use_residual=True)


    def forward(self, batch: Batch) -> Batch:
        """
        Forward pass of the GNNBlockLayer.

        Args:
            batch (Batch): The input batch of data.

        Returns:
            Batch: The output batch after processing.
        """
        # Register input for residual connection
        self.mpnn_residual.register(batch.x)

        # Message passing
        batch = self.mpnn(batch)

        # Optional normalization
        batch.x = normalize(batch.x, batch, self.mpnn_norm)
        # Activation
        batch.x = self.activation(batch.x)

        # Optional dropout
        if self.dropout is not None:
            batch.x = self.dropout(batch.x)

        # Optional residual connection
        batch.x = self.mpnn_residual.apply(batch.x)

        # Optional Feed-Forward Network (FFN)
        if self.use_ffn:
            self.ffn_residual.register(batch.x)
            batch.x = normalize(batch.x, batch, self.ffn_norm_in)
            batch.x = self.ffn(batch.x)
            batch.x = self.dropout(batch.x)
            batch.x = self.ffn_residual.apply(batch.x)
            batch.x = normalize(batch.x, batch, self.ffn_norm_out)

        return batch


    def __init_subclass__(cls) -> None:
        enforce_base_init(GNNBlockLayer)(cls)
        return super().__init_subclass__()


