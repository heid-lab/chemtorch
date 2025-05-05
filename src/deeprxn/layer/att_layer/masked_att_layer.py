from typing import Literal, Optional, Tuple

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.utils import to_dense_batch

from deeprxn.layer.att_layer.att_layer_base import AttLayer
from deeprxn.representation.reaction_graph import AtomOriginType


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

        if self.mode == "qm" or self.mode == "qm_inter" or self.mode == "qm_local":
            qm_dense, qm_mask = to_dense_batch(batch.qm_f, batch.batch, fill_value=0)

        if self.mode == "atom" or self.mode == "atom_inter" or self.mode == "conjugated" or self.mode == "conjugated_inter":
            single_f_dense, _ = to_dense_batch(batch.single_f, batch.batch, fill_value=-1)

        batch_size, max_nodes, _ = x_dense.size()

        if self.mode in [
            "intra",
            "inter",
            "compound",
            "local_inter",
            "local_compound_inter",
            "local_compound",
            "same_compound",
            "other_compounds_same_category",
            "qm_inter",
            "atom_inter",
            "conjugated_inter",
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

        if self.mode == "local" or self.mode == "qm_local":
            # Get dimensions and device.
            batch_size, max_nodes, _ = x_dense.size()
            device = x_dense.device

            # Compute number of valid nodes per batch (mask is bool with True for valid nodes)
            # and create cumulative counts to determine the starting index of each batch in the global node list.
            counts = mask.sum(dim=1).long()  # shape: [batch_size]
            cum_counts = torch.cat(
                [torch.tensor([0], device=device), counts.cumsum(dim=0)]
            )  # shape: [batch_size+1]

            # Create a mapping from global node index to its local index within its batch.
            global_node_indices = torch.arange(batch.x.size(0), device=device)
            # local index = global index - starting index of its batch.
            local_indices = global_node_indices - cum_counts[batch.batch]

            # For each edge in the global edge index, compute its batch id and local indices.
            src_global = batch.edge_index[0]
            tgt_global = batch.edge_index[1]
            edge_batch = batch.batch[
                src_global
            ]  # each edge is assumed to be intra-batch

            local_src = local_indices[src_global]
            local_tgt = local_indices[tgt_global]

            # Set the corresponding entries in the attention mask to True for both directions.
            attention_mask[edge_batch, local_src, local_tgt] = True
            attention_mask[edge_batch, local_tgt, local_src] = True

            # Vectorized diagonal setting: create a diagonal mask for each batch that covers the valid nodes.
            diag_indices = torch.arange(max_nodes, device=device).unsqueeze(
                0
            )  # shape: [1, max_nodes]
            valid_diag = diag_indices < counts.unsqueeze(
                1
            )  # shape: [batch_size, max_nodes]
            diag_eye = (
                torch.eye(max_nodes, dtype=torch.bool, device=device)
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
            )
            attention_mask = attention_mask | (
                diag_eye & valid_diag.unsqueeze(-1)
            )

        if self.mode == "atom" or self.mode == "conjugated":
            self_attention = torch.eye(max_nodes, dtype=torch.bool, device=device)
            self_attention = self_attention.unsqueeze(0).expand(batch_size, -1, -1)
            self_attention = self_attention & mask.unsqueeze(-1)
            same_type_mask = (single_f_dense.unsqueeze(2) == single_f_dense.unsqueeze(1))
            
            if same_type_mask.dim() > 3:
                same_type_mask = same_type_mask.all(dim=-1)
            
            valid_nodes = mask.unsqueeze(-1) & mask.unsqueeze(1)
            same_type_attention = same_type_mask & valid_nodes
            
            attention_mask = attention_mask | same_type_attention

        elif self.mode == "atom_inter" or self.mode == "conjugated_inter": # double check
            self_attention = torch.eye(max_nodes, dtype=torch.bool, device=device)
            self_attention = self_attention.unsqueeze(0).expand(batch_size, -1, -1)
            self_attention = self_attention & mask.unsqueeze(-1)
            
            attention_mask = attention_mask | self_attention
            
            same_type_mask = (single_f_dense.unsqueeze(2) == single_f_dense.unsqueeze(1))
            
            if same_type_mask.dim() > 3:
                same_type_mask = same_type_mask.all(dim=-1)
            
            valid_nodes = mask.unsqueeze(-1) & mask.unsqueeze(1)
            
            reactant_reactant = is_reactant.unsqueeze(-1) & is_reactant.unsqueeze(1)
            product_product = is_product.unsqueeze(-1) & is_product.unsqueeze(1)
            
            same_type_reactant_reactant = reactant_reactant & same_type_mask & valid_nodes
            
            same_type_product_product = product_product & same_type_mask & valid_nodes
            
            attention_mask = attention_mask | same_type_reactant_reactant | same_type_product_product

        if self.mode == "inter" or self.mode == "qm_inter":
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

        elif self.mode == "same_compound":  # TODO : Double check this
            compound_idx_dense, _ = to_dense_batch(
                batch.atom_compound_idx, batch.batch, fill_value=-1
            )

            same_compound_attention = compound_idx_dense.unsqueeze(
                -1
            ) == compound_idx_dense.unsqueeze(1)

            valid_nodes = mask.unsqueeze(-1) & mask.unsqueeze(1)
            same_compound_attention = same_compound_attention & valid_nodes

            attention_mask = attention_mask | same_compound_attention

        elif self.mode == "other_compounds_same_category":
            # Self-attention (each node attends to itself)
            self_attention = torch.eye(
                max_nodes, dtype=torch.bool, device=device
            )
            self_attention = self_attention.unsqueeze(0).expand(
                batch_size, -1, -1
            )
            self_attention = self_attention & mask.unsqueeze(-1)

            # Get compound indices and create different compound mask
            compound_idx_dense, _ = to_dense_batch(
                batch.atom_compound_idx, batch.batch, fill_value=-1
            )
            different_compound_attention = compound_idx_dense.unsqueeze(
                -1
            ) != compound_idx_dense.unsqueeze(1)

            # Create same category mask (reactant-reactant or product-product)
            reactant_reactant = is_reactant.unsqueeze(
                -1
            ) & is_reactant.unsqueeze(1)
            product_product = is_product.unsqueeze(-1) & is_product.unsqueeze(
                1
            )
            same_category = reactant_reactant | product_product

            # Valid nodes only
            valid_nodes = mask.unsqueeze(-1) & mask.unsqueeze(1)

            # Combine: different compounds but same category
            different_compound_same_category = (
                different_compound_attention & same_category & valid_nodes
            )

            # Final attention mask
            attention_mask = (
                attention_mask
                | self_attention
                | different_compound_same_category
            )

        if self.mode == "self" or self.mode == "qm":
            attention_mask = None
        else:
            attention_mask = attention_mask.repeat_interleave(
                self.num_heads, dim=0
            )
            attention_mask = ~attention_mask

        if self.mode == "qm" or self.mode == "qm_inter" or self.mode == "qm_local":
            att_output, _ = self.attention(
                qm_dense,
                qm_dense,
                x_dense,
                # key_padding_mask=~mask, TODO: WHY NAN ????????????????????????????
                attn_mask=attention_mask,
                need_weights=False,
            )
        else:
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
