from typing import Any, Callable, Dict, Optional, Union
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn.resolver import activation_resolver

from deepreaction.layer.utils import ResidualConnection, init_2_layer_ffn, init_dropout, init_norm, normalize


class GPSBlock(nn.Module):
    """
    Graph Propagation and Self-Attention (GPS) Block for Graph Neural Networks.
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
        graph_conv: nn.Module,
        attention: nn.MultiheadAttention,
        residual: bool = False,
        dropout = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        hidden_channels: int = None,
    ):
        """
        Initializes the GPSBlock.
        
        Args:
            graph_conv (nn.Module): The graph convolution.
            attention (nn.MultiheadAttention): The attention layer.
            residual (bool): Whether to use a residual connection for the graph convolution (default: `False`).
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
            hidden_channels (int): Number of hidden channels (needs to be specified 
                if a norm is used. Default: `None`).

        Raises:
            ValueError: If `dropout` is less than `0`.
        """
        super(GPSBlock, self).__init__()
        self.activation = activation_resolver(act, **(act_kwargs or {}))
        self.dropout = init_dropout(dropout)

        self.graph_conv = graph_conv
        self.norm = init_norm(norm, hidden_channels, norm_kwargs)
        self.residual = ResidualConnection(residual)

        self.attn = attention
        self.attn_residual = ResidualConnection(use_residual=True)
        self.attn_norm = init_norm(norm, hidden_channels, norm_kwargs)

        self.ffn = init_2_layer_ffn(hidden_channels, dropout, self.activation)
        self.ffn_norm = init_norm(norm, hidden_channels, norm_kwargs)
        self.ffn_residual = ResidualConnection(use_residual=True)


    def forward(self, batch: Batch):
        # Register original input features for residual connection
        self.residual.register(batch.x)
        self.ffn_residual.register(batch.x)
        self.attn_residual.register(batch.x)

        x_dense, mask = to_dense_batch(batch.x, batch.batch)

        # MPNN
        batch = self.graph_conv(batch)
        h_mpnn = batch.x
        h_mpnn = self.residual.apply(h_mpnn)
        h_mpnn = normalize(h_mpnn, batch, self.norm)

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
