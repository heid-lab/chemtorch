from typing import Literal, Optional, Tuple

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.utils import to_dense_batch

from deeprxn.layer.att_layer.att_layer_base import AttLayer
from deeprxn.representation.rxn_graph_base import AtomOriginType


class DeepRXNAttLayer(AttLayer):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int = 1,
        dropout: float = 0.0,
        batch_first: bool = True,
        layer_norm: bool = False,
        batch_norm: bool = True,
    ):
        AttLayer.__init__(self, in_channels, out_channels)
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        self.attention = nn.MultiheadAttention(
            embed_dim=out_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first,  # data [num_batches, max_nodes_per_batch, hidden_size]
        )

        if self.layer_norm and self.batch_norm:
            raise ValueError("Cannot use both layer and batch normalization")

        if self.layer_norm:
            self.ln = LayerNorm(out_channels)

        if self.batch_norm:
            self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, batch: Batch) -> Batch:
        reactant_mask = batch.atom_origin_type == AtomOriginType.REACTANT
        product_mask = batch.atom_origin_type == AtomOriginType.PRODUCT

        reactant_features = batch.x[reactant_mask]
        product_features = batch.x[product_mask]
        reactant_batch = batch.batch[reactant_mask]
        product_batch = batch.batch[product_mask]

        reactant_features_dense, reactant_mask_dense = to_dense_batch(
            reactant_features, reactant_batch
        )
        product_features_dense, product_mask_dense = to_dense_batch(
            product_features, product_batch
        )

        updated_reactants, _ = self.attention(
            reactant_features_dense,
            reactant_features_dense,
            reactant_features_dense,
            key_padding_mask=~reactant_mask_dense,
            need_weights=False,
        )

        updated_products, _ = self.attention(
            product_features_dense,
            product_features_dense,
            product_features_dense,
            key_padding_mask=~product_mask_dense,
            need_weights=False,
        )

        updated_reactants = updated_reactants[reactant_mask_dense]
        updated_products = updated_products[product_mask_dense]

        h_attn_reactants = batch.x[reactant_mask] + updated_reactants
        h_attn_products = batch.x[product_mask] + updated_products

        combined_features = batch.x.clone()
        combined_features[reactant_mask] = h_attn_reactants
        combined_features[product_mask] = h_attn_products

        combined_features_dense, combined_mask_dense = to_dense_batch(
            combined_features, batch.batch
        )

        final_features, _ = self.attention(
            combined_features_dense,
            combined_features_dense,
            combined_features_dense,
            need_weights=False,
        )

        final_features = final_features[combined_mask_dense]
        batch.h_attn = combined_features + final_features

        if self.layer_norm:
            batch.h_attn = self.ln(batch.h_attn, batch.batch)
        if self.batch_norm:
            batch.h_attn = self.bn(batch.h_attn)

        return batch
