from typing import Literal, Optional, Tuple

import torch
from torch import nn
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.utils import to_dense_batch


class MessageAggregationAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 1,
        dropout: float = 0.02,
        layer_norm: bool = False,
        batch_norm: bool = False,
    ):
        super(MessageAggregationAttention, self).__init__()
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        self.attention = nn.MultiheadAttention(
            hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        if self.layer_norm and batch_norm:
            raise ValueError("Cannot use both layer and batch normalization")

        if self.layer_norm:
            self.ln = LayerNorm(hidden_size)

        if self.batch_norm:
            self.bn = nn.BatchNorm1d(hidden_size)

        self.ffn_dropout_1 = nn.Dropout(dropout)
        self.ffn_linear_1 = nn.Linear(hidden_size, hidden_size * 2)
        self.ffn_activation = nn.ReLU()
        self.ffn_dropout_2 = nn.Dropout(dropout)
        self.ffn_linear_2 = nn.Linear(hidden_size * 2, hidden_size)
        if self.layer_norm:
            self.ffn_ln = LayerNorm(hidden_size)
        if self.batch_norm:
            self.ffn_bn = nn.BatchNorm1d(hidden_size)

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        incoming_edges_list: Optional[torch.Tensor] = None,
        incoming_edges_batch: Optional[torch.Tensor] = None,
        edge_batch: Optional[torch.Tensor] = None,
        incoming_edges_batch_from_zero: Optional[torch.Tensor] = None,
        edge_batch_2: Optional[torch.Tensor] = None,
        is_real_bond: Optional[torch.Tensor] = None,
        edge_to_node: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
        debug: bool = False,
        lol: bool = False,
    ):
        row, col = edge_index

        edge_attr_dense, mask_edge_attr_dense = to_dense_batch(
            edge_attr,
            batch=edge_batch,
        )

        edge_attr_rearranged = edge_attr[incoming_edges_list]

        incoming_edges, mask = to_dense_batch(
            edge_attr_rearranged,
            batch=incoming_edges_batch,
        )

        edge_attr_dense_updated, attention_weights = self.attention(
            edge_attr_dense,
            incoming_edges,
            incoming_edges,
            key_padding_mask=~mask,
            need_weights=return_attention_weights,
        )

        edge_attr_dense_updated = edge_attr_dense_updated[mask_edge_attr_dense]

        att_output = edge_attr + edge_attr_dense_updated

        if self.layer_norm:
            att_output = self.ln(
                att_output, edge_batch
            )  # look into what batch to pass here

        if self.batch_norm:
            att_output = self.bn(att_output)

        # feed-forward network
        att_output_ffn = self.ffn_dropout_1(att_output)
        att_output_ffn = self.ffn_linear_1(att_output_ffn)
        att_output_ffn = self.ffn_activation(att_output_ffn)
        att_output_ffn = self.ffn_dropout_2(att_output_ffn)
        att_output_ffn = self.ffn_linear_2(att_output_ffn)

        # residual connection
        att_output = att_output + att_output_ffn

        if self.layer_norm:
            att_output = self.ffn_ln(att_output)

        if self.batch_norm:
            att_output = self.ffn_bn(att_output)

        return att_output
