from typing import Literal, Optional, Tuple

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.utils import to_dense_batch

from deeprxn.layer.att_layer.att_layer_base import AttLayer
from deeprxn.representation.rxn_graph_base import AtomOriginType


class ReactantProductAttLayer(AttLayer):
    """
    Attention layer for reactants and products in reaction graphs.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_size: int,
        attention: Literal[
            "reactants", "products", "reactants_products"
        ] = "reactants_products",
        num_heads: int = 1,
        dropout: float = 0.02,
        layer_norm: bool = False,
        batch_norm: bool = False,
    ):
        AttLayer.__init__(self, in_channels, out_channels)
        self.hidden_size = hidden_size
        self.attention = attention
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        if attention in ["reactants", "reactants_products"]:
            self.attention_reactants = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,  # data [num_batches, max_nodes_per_batch, hidden_size]
            )
        if attention in ["products", "reactants_products"]:
            self.attention_products = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )

        if self.layer_norm and self.batch_norm:
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

    def forward(self, batch: Batch) -> Batch:
        """
        Forward pass of the ReactantProductAttention layer.

        Args:
            node_features: Node features of shape [num_nodes (in whole batch), hidden_size]
            atom_origin_type: Tensor indicating the origin type of each atom [num_nodes (in whole batch)]
            batch: Batch assignment for each node

        Returns:
            Updated node features
        """
        reactant_mask = batch.atom_origin_type == AtomOriginType.REACTANT
        product_mask = batch.atom_origin_type == AtomOriginType.PRODUCT

        # features and batch indices for reactants and products
        reactant_features = batch.x[reactant_mask]
        product_features = batch.x[product_mask]
        reactant_batch = batch.batch[reactant_mask]
        product_batch = batch.batch[product_mask]

        # to_dense_batch
        # [num_batches, max_nodes_per_batch, hidden_size], [num_batches, max_nodes_per_batch]
        # the mask indicates for each batch which nodes are real and which are padding
        reactant_features_dense, reactant_mask_dense = to_dense_batch(
            reactant_features, reactant_batch
        )
        product_features_dense, product_mask_dense = to_dense_batch(
            product_features, product_batch
        )

        if self.attention in ["reactants", "reactants_products"]:
            # Q: reactants, K: products, V: products
            updated_reactants, _ = self.attention_reactants(
                reactant_features_dense,
                product_features_dense,
                product_features_dense,
                key_padding_mask=~product_mask_dense,
                need_weights=False,
            )
        else:
            updated_reactants = reactant_features_dense

        if self.attention in ["products", "reactants_products"]:
            # Q: products, K: reactants, V: reactants
            updated_products, _ = self.attention_products(
                product_features_dense,
                reactant_features_dense,
                reactant_features_dense,
                key_padding_mask=~reactant_mask_dense,
                need_weights=False,
            )
        else:
            updated_products = product_features_dense

        # unpad
        updated_reactants = updated_reactants[reactant_mask_dense]
        updated_products = updated_products[product_mask_dense]

        # TODO: look into better solution, gave gradient error
        att_output = batch.x.clone()

        # residual connection
        if self.attention in ["reactants", "reactants_products"]:
            att_output[reactant_mask] = (
                batch.x[reactant_mask] + updated_reactants
            )

        if self.attention in ["products", "reactants_products"]:
            att_output[product_mask] = batch.x[product_mask] + updated_products

        # TODO: do we need different layer norm for reactants and products?
        if self.layer_norm:
            att_output = self.ln(att_output, batch.batch)

        if self.batch_norm:
            att_output = self.bn(att_output)

        # feed-forward network
        att_output_ffn = self.ffn_dropout_1(att_output)
        att_output_ffn = self.ffn_linear_1(att_output_ffn)
        att_output_ffn = self.ffn_activation(att_output_ffn)
        att_output_ffn = self.ffn_dropout_2(att_output_ffn)
        att_output_ffn = self.ffn_linear_2(att_output_ffn)

        # residual connection
        batch.x = att_output + att_output_ffn

        if self.layer_norm:
            batch.x = self.ffn_ln(batch.x)

        if self.batch_norm:
            batch.x = self.ffn_bn(batch.x)

        return batch
