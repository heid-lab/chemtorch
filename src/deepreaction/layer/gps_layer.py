from typing import Any, Callable, Dict, Optional, Union
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn.resolver import activation_resolver

from deepreaction.layer.gnn_block import GNNBlockLayer
from deepreaction.layer.utils import ResidualConnection, init_2_layer_ffn, init_dropout, init_norm, normalize


class GPSLayer(nn.Module):
    """
    Graph Propagation and Self-Attention Layer (GPSLayer) for Graph Neural Networks.
    This layer combines message passing neural networks (MPNN) with self-attention
    mechanisms to capture both local and global information in graph-structured data.

    Citation:
    Rampášek, L., Galkin, M., Dwivedi, V. P., Luu, A. T., Wolf, G., & 
    Beaini, D. (2022). Recipe for a general, powerful, scalable graph 
    transformer. Advances in Neural Information Processing Systems, 35, 
    14501-14515. https://arxiv.org/abs/2205.12454
    """
    def __init__(
        self,
        mpnn: nn.Module,
        attention: nn.MultiheadAttention,
        hidden_channels: int,
        dropout = 0.0,
        use_mpnn_residual: bool = False,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,  # TODO: Add suppoert for PNA 'degree_statistics'
    ):
        """
        Initializes the GPSLayer.
        
        Args:
            mpnn (nn.Module): The message passing neural network (MPNN) layer.
            attention (nn.MultiheadAttention): The attention layer.
            hidden_channels (int): Number of hidden channels.
            use_mpnn_residual (bool): Whether to use a residual connection for the MPNN (default: `False`).
            dropout (float): Dropout rate (default: `0.0`).
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
            ValueError: If `dropout` is less than `0`.
        """
        self.activation = activation_resolver(act, **(act_kwargs or {}))
        self.dropout = init_dropout(dropout)

        self.mpnn = mpnn
        self.mpnn_norm = init_norm(norm, hidden_channels, norm_kwargs)
        self.mpnn_residual = ResidualConnection(use_mpnn_residual)

        self.attn = attention
        self.attn_residual = ResidualConnection(use_residual=True)
        self.attn_norm = init_norm(norm, hidden_channels, norm_kwargs)

        self.ffn = init_2_layer_ffn(hidden_channels, dropout, self.activation)
        self.ffn_norm = init_norm(norm, hidden_channels, norm_kwargs)
        self.ffn_residual = ResidualConnection(use_residual=True)


    def forward(self, batch: Batch):
        # Register original input features for residual connection
        self.mpnn_residual.register(batch.x)
        self.ffn_residual.register(batch.x)
        self.attn_residual.register(batch.x)

        x_dense, mask = to_dense_batch(batch.x, batch.batch)

        # MPNN
        batch = self.mpnn(batch)
        h_mpnn = batch.x
        h_mpnn = self.mpnn_residual.apply(h_mpnn)
        h_mpnn = normalize(h_mpnn, batch, self.mpnn_norm)

        # Self-attention
        h_attn = self.attn(
            x_dense,
            x_dense,
            x_dense,
            attn_mask=None,
            key_padding_mask=~mask,
            need_weights=False,
        )[0][mask]
        h_attn = self.dropout(h_attn)
        h_attn = self.attn_residual.apply(h_attn)
        h_attn = normalize(h_attn, batch, self.attn_norm)

        # Combine local MPNN and global self-attention features
        h = h_mpnn + h_attn

        # Feed Forward block.
        h = self.ffn(h)
        h = self.dropout(h)
        h = self.ffn_residual.apply(h)
        h = normalize(h, batch, self.ffn_norm)
        batch.x = h
        return batch
