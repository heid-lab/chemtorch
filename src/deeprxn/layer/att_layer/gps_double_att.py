from typing import Literal, Optional, Tuple

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.utils import to_dense_batch

from deeprxn.layer.att_layer.att_layer_base import AttLayer
from deeprxn.representation.rxn_graph_base import AtomOriginType


class GPSDoubleAttLayer(AttLayer):
    """
    Attention layer for reactants and products in reaction graphs.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_size: int,
        num_heads: int = 1,
        dropout: float = 0.02,
        layer_norm: bool = False,
        batch_norm: bool = False,
        with_nn: bool = False,
    ):
        AttLayer.__init__(self, in_channels, out_channels)
        self.hidden_size = hidden_size
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.with_nn = with_nn

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,  # data [num_batches, max_nodes_per_batch, hidden_size]
        )

        if self.layer_norm and self.batch_norm:
            raise ValueError("Cannot use both layer and batch normalization")

        if self.layer_norm:
            self.ln = LayerNorm(hidden_size)

        if self.batch_norm:
            self.bn = nn.BatchNorm1d(hidden_size)

        if with_nn:
            self.ffn_dropout_1 = nn.Dropout(dropout)
            self.ffn_linear_1 = nn.Linear(hidden_size, hidden_size * 2)
            self.ffn_activation = nn.ReLU()
            self.ffn_dropout_2 = nn.Dropout(dropout)
            self.ffn_linear_2 = nn.Linear(hidden_size * 2, hidden_size)
            if self.layer_norm:
                self.ffn_ln = LayerNorm(hidden_size)
            if self.batch_norm:
                self.ffn_bn = nn.BatchNorm1d(hidden_size)

    def forward(self, batch_1, batch_2s) -> Batch:

        features_dense_1, mask_dense_1 = to_dense_batch(
            batch_1.x, batch_1.batch
        )
        features_dense_2, mask_dense_2 = to_dense_batch(
            batch_2s.x, batch_2s.batch
        )

        updated_1, _ = self.attention(
            features_dense_1,
            features_dense_2,
            features_dense_2,
            key_padding_mask=~mask_dense_2,
            need_weights=False,
        )

        updated_1 = updated_1[mask_dense_1]
        att_output = batch_1.x + updated_1

        if self.layer_norm:
            att_output = self.ln(att_output, batch_1.batch)

        if self.batch_norm:
            att_output = self.bn(att_output)

        if self.with_nn:
            # feed-forward network
            att_output_ffn = self.ffn_dropout_1(att_output)
            att_output_ffn = self.ffn_linear_1(att_output_ffn)
            att_output_ffn = self.ffn_activation(att_output_ffn)
            att_output_ffn = self.ffn_dropout_2(att_output_ffn)
            att_output_ffn = self.ffn_linear_2(att_output_ffn)

            batch_1.x = att_output + att_output_ffn

            if self.layer_norm:
                batch_1.x = self.ffn_ln(batch_1.x)

            if self.batch_norm:
                batch_1.x = self.ffn_bn(batch_1.x)
        else:
            batch_1.x = att_output

        return batch_1
