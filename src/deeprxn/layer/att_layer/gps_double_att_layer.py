from typing import Literal, Optional, Tuple

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.utils import to_dense_batch

from deeprxn.layer.att_layer.att_layer_base import AttLayer
from deeprxn.representation.reaction_graph import AtomOriginType


class ReactantProductAttLayer(AttLayer):
    """
    Attention layer for reactants and products in reaction graphs.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        attention: Literal[
            "reactants", "products", "reactants_products"
        ] = "reactants_products",
        num_heads: int = 1,
        dropout: float = 0.0,
        batch_first: bool = True,
        layer_norm: bool = False,
        batch_norm: bool = True,
    ):
        AttLayer.__init__(self)
        self.attention = attention
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        if attention in ["reactants", "reactants_products"]:
            self.attention_reactants = nn.MultiheadAttention(
                embed_dim=out_channels,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=batch_first,  # data [num_batches, max_nodes_per_batch, hidden_size]
            )
        if attention in ["products", "reactants_products"]:
            self.attention_products = nn.MultiheadAttention(
                embed_dim=out_channels,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )

        if self.layer_norm and self.batch_norm:
            raise ValueError("Cannot use both layer and batch normalization")

        if self.layer_norm:
            self.ln = LayerNorm(out_channels)

        if self.batch_norm:
            self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, h_dense, mask, h_dense_2, mask_2) -> Batch:

        # Q: reactants, K: products, V: products
        updated, _ = self.attention(
            h_dense,
            h_dense_2,
            h_dense_2,
            key_padding_mask=~mask_2,
            need_weights=False,
        )

        # unpad
        updated = updated[mask]

        return updated
