from typing import Literal, Optional, Tuple

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.utils import to_dense_batch

from deeprxn.layer.att_layer.att_layer_base import AttLayer
from deeprxn.representation.rxn_graph_base import AtomOriginType


class MaskedAttLayer(AttLayer):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        dropout: float = 0.02,
        batch_first: bool = True,
        mode: Literal["self", "local", "intra", "inter"] = "local",
        layer_norm: bool = False,
        batch_norm: bool = False,
    ):
        AttLayer.__init__(self)
        self.mode = mode
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.num_heads = num_heads

        if self.layer_norm and self.batch_norm:
            raise ValueError("Cannot use both layer and batch normalization")

        if self.layer_norm:
            self.pre_ln = LayerNorm(embed_dim)
        if self.batch_norm:
            self.pre_bn = nn.BatchNorm1d(embed_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,  # data [num_batches, max_nodes_per_batch, hidden_size]
        )

        if self.layer_norm:
            self.ln = LayerNorm(embed_dim)

        if self.batch_norm:
            self.bn = nn.BatchNorm1d(embed_dim)

        self.ffn_dropout_1 = nn.Dropout(dropout)
        self.ffn_linear_1 = nn.Linear(embed_dim, embed_dim * 2)
        self.ffn_activation = nn.ReLU()  # nn.SiLU()  # nn.ReLU()
        self.ffn_dropout_2 = nn.Dropout(dropout)
        self.ffn_linear_2 = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, batch: Batch) -> Batch:

        if self.layer_norm:
            batch.x = self.pre_ln(batch.x, batch.batch)
        if self.batch_norm:
            batch.x = self.pre_bn(batch.x)

        device = batch.x.device

        x_dense, mask = to_dense_batch(batch.x, batch.batch, fill_value=0)
        atom_types_dense, _ = to_dense_batch(
            batch.atom_origin_type, batch.batch, fill_value=-1
        )

        batch_size, max_nodes, _ = x_dense.size()

        if self.mode in [
            "intra",
            "inter",
            "compound",
            "local_inter",
            "local_compound_inter",
            "local_compound",
        ]:
            is_reactant = atom_types_dense == AtomOriginType.REACTANT.value
            is_product = atom_types_dense == AtomOriginType.PRODUCT.value

        attention_mask = torch.zeros(
            (batch_size, max_nodes, max_nodes),
            dtype=torch.bool,
            device=device,
        )
        attention_mask.diagonal(dim1=1, dim2=2)[:] = (
            ~mask
        )  # this is important due to bug with padding

        if self.mode == "local":  # TODO: check to make more efficient
            for b in range(batch_size):
                batch_mask = batch.batch == b
                batch_nodes = batch_mask.nonzero().squeeze()

                edge_mask = batch_mask[batch.edge_index[0]]
                batch_edges = batch.edge_index[:, edge_mask]

                local_edges = batch_edges - batch_nodes[0]

                attention_mask[b, local_edges[0], local_edges[1]] = True
                attention_mask[b, local_edges[1], local_edges[0]] = True

                attention_mask[b].diagonal()[: mask[b].sum()] = True

        if self.mode == "inter":
            attention_mask = torch.logical_or(
                attention_mask,
                is_reactant.unsqueeze(-1) & is_product.unsqueeze(1),
            )
            attention_mask = torch.logical_or(
                attention_mask,
                is_product.unsqueeze(-1) & is_reactant.unsqueeze(1),
            )

        if self.mode == "intra":
            structure_attention = (
                is_reactant.unsqueeze(-1) & is_reactant.unsqueeze(1)
            ) | (is_product.unsqueeze(-1) & is_product.unsqueeze(1))

            attention_mask = attention_mask | structure_attention

        if self.mode == "compound":
            compound_idx_dense, _ = to_dense_batch(
                batch.atom_compound_idx, batch.batch, fill_value=-1
            )

            compound_attention = compound_idx_dense.unsqueeze(
                -1
            ) != compound_idx_dense.unsqueeze(1)

            self_attention = torch.eye(
                max_nodes, dtype=torch.bool, device=device
            )
            self_attention = self_attention.unsqueeze(0).expand(
                batch_size, -1, -1
            )
            self_attention = self_attention & mask.unsqueeze(-1)
            structure_attention = (
                is_reactant.unsqueeze(-1) & is_reactant.unsqueeze(1)
            ) | (is_product.unsqueeze(-1) & is_product.unsqueeze(1))

            attention_mask = (
                attention_mask
                | (structure_attention & compound_attention)
                | self_attention
            )

        if self.mode == "local_inter":
            for b in range(batch_size):
                batch_mask = batch.batch == b
                batch_nodes = batch_mask.nonzero().squeeze()

                edge_mask = batch_mask[batch.edge_index[0]]
                batch_edges = batch.edge_index[:, edge_mask]

                local_edges = batch_edges - batch_nodes[0]

                attention_mask[b, local_edges[0], local_edges[1]] = True
                attention_mask[b, local_edges[1], local_edges[0]] = True

                attention_mask[b].diagonal()[: mask[b].sum()] = True

            attention_mask = torch.logical_or(
                attention_mask,
                is_reactant.unsqueeze(-1) & is_product.unsqueeze(1),
            )
            attention_mask = torch.logical_or(
                attention_mask,
                is_product.unsqueeze(-1) & is_reactant.unsqueeze(1),
            )

        if self.mode == "local_inter":
            for b in range(batch_size):
                batch_mask = batch.batch == b
                batch_nodes = batch_mask.nonzero().squeeze()

                edge_mask = batch_mask[batch.edge_index[0]]
                batch_edges = batch.edge_index[:, edge_mask]

                local_edges = batch_edges - batch_nodes[0]

                attention_mask[b, local_edges[0], local_edges[1]] = True
                attention_mask[b, local_edges[1], local_edges[0]] = True

                attention_mask[b].diagonal()[: mask[b].sum()] = True

            attention_mask = torch.logical_or(
                attention_mask,
                is_reactant.unsqueeze(-1) & is_product.unsqueeze(1),
            )
            attention_mask = torch.logical_or(
                attention_mask,
                is_product.unsqueeze(-1) & is_reactant.unsqueeze(1),
            )

        if self.mode == "local_compound":
            for b in range(batch_size):
                batch_mask = batch.batch == b
                batch_nodes = batch_mask.nonzero().squeeze()

                edge_mask = batch_mask[batch.edge_index[0]]
                batch_edges = batch.edge_index[:, edge_mask]

                local_edges = batch_edges - batch_nodes[0]

                attention_mask[b, local_edges[0], local_edges[1]] = True
                attention_mask[b, local_edges[1], local_edges[0]] = True

                attention_mask[b].diagonal()[: mask[b].sum()] = True

            compound_idx_dense, _ = to_dense_batch(
                batch.atom_compound_idx, batch.batch, fill_value=-1
            )
            compound_attention = compound_idx_dense.unsqueeze(
                -1
            ) != compound_idx_dense.unsqueeze(1)

            self_attention = torch.eye(
                max_nodes, dtype=torch.bool, device=device
            )
            self_attention = self_attention.unsqueeze(0).expand(
                batch_size, -1, -1
            )
            self_attention = self_attention & mask.unsqueeze(-1)

            structure_attention = (
                is_reactant.unsqueeze(-1) & is_reactant.unsqueeze(1)
            ) | (is_product.unsqueeze(-1) & is_product.unsqueeze(1))

            attention_mask = (
                attention_mask
                | (structure_attention & compound_attention)
                | self_attention
            )

        if self.mode == "local_compound_inter":
            for b in range(batch_size):
                batch_mask = batch.batch == b
                batch_nodes = batch_mask.nonzero().squeeze()

                edge_mask = batch_mask[batch.edge_index[0]]
                batch_edges = batch.edge_index[:, edge_mask]

                local_edges = batch_edges - batch_nodes[0]

                attention_mask[b, local_edges[0], local_edges[1]] = True
                attention_mask[b, local_edges[1], local_edges[0]] = True

                attention_mask[b].diagonal()[: mask[b].sum()] = True

            compound_idx_dense, _ = to_dense_batch(
                batch.atom_compound_idx, batch.batch, fill_value=-1
            )
            compound_attention = compound_idx_dense.unsqueeze(
                -1
            ) != compound_idx_dense.unsqueeze(1)

            self_attention = torch.eye(
                max_nodes, dtype=torch.bool, device=device
            )
            self_attention = self_attention.unsqueeze(0).expand(
                batch_size, -1, -1
            )
            self_attention = self_attention & mask.unsqueeze(-1)

            structure_attention = (
                is_reactant.unsqueeze(-1) & is_reactant.unsqueeze(1)
            ) | (is_product.unsqueeze(-1) & is_product.unsqueeze(1))

            inter_attention = torch.logical_or(
                is_reactant.unsqueeze(-1) & is_product.unsqueeze(1),
                is_product.unsqueeze(-1) & is_reactant.unsqueeze(1),
            )

            attention_mask = (
                attention_mask
                | (structure_attention & compound_attention)
                | self_attention
                | inter_attention
            )

        if self.mode == "self":
            attention_mask = None
        else:
            attention_mask = attention_mask.repeat_interleave(
                self.num_heads, dim=0
            )
            attention_mask = ~attention_mask

        att_output, _ = self.attention(
            x_dense,
            x_dense,
            x_dense,
            # key_padding_mask=~mask, TODO: WHY NAN ????????????????????????????
            attn_mask=attention_mask,
            need_weights=False,
        )

        att_output = att_output[mask] + batch.x

        if self.layer_norm:
            att_output_normed = self.ln(att_output, batch.batch)
        if self.batch_norm:
            att_output_normed = self.bn(att_output)

        ffn_output = self.ffn_dropout_1(att_output_normed)
        ffn_output = self.ffn_linear_1(ffn_output)
        ffn_output = self.ffn_activation(ffn_output)
        ffn_output = self.ffn_dropout_2(ffn_output)
        ffn_output = self.ffn_linear_2(ffn_output)

        batch.x = ffn_output + att_output_normed

        return batch
