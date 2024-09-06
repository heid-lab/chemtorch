from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn
from torch_geometric.nn import global_add_pool
from torch_scatter import scatter_add

from deeprxn.data import AtomOriginType
from deeprxn.layers import DMPNNConv, TransConvLayer


class GNN(nn.Module):
    """
    Graph Neural Network for molecular property prediction.
    This model uses DMPNN convolutions and supports various pooling methods.
    Option for attention mechanism is also available.
    """

    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        depth: int,
        hidden_size: int,
        dropout: float,
        layer_cfg: DictConfig,
        attention: Literal[
            None, "reactants", "products", "reactants_products"
        ] = None,
        pool_type: Literal[
            "global", "reactants", "products", "dummy"
        ] = "global",
        pool_real_only: bool = False,  # TODO: look into this
        double_features: bool = False,
        fast_attention: bool = True,
    ):
        super(GNN, self).__init__()
        self.depth = depth
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.pool_type = pool_type
        self.separate_nn = layer_cfg.separate_nn
        self.pool_real_only = pool_real_only
        self.double_features = double_features and attention is not None
        self.attention = attention
        self.fast_attention = fast_attention

        self.edge_init = nn.Linear(
            num_node_features + num_edge_features, self.hidden_size
        )
        self.convs = torch.nn.ModuleList()
        for _ in range(self.depth):
            self.convs.append(DMPNNConv(self.hidden_size, self.separate_nn))
        self.edge_to_node = nn.Linear(
            num_node_features + self.hidden_size, self.hidden_size
        )

        ffn_input_size = (
            self.hidden_size * 2 if self.double_features else self.hidden_size
        )
        self.ffn = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(ffn_input_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 1),
        )

        if self.fast_attention:  # TODO: maybe make more compact
            if (
                self.attention == "reactants"
                or self.attention == "reactants_products"
            ):
                self.attention_reactants = TransConvLayer(
                    layer_cfg.in_channels,
                    layer_cfg.out_channels,
                    layer_cfg.num_heads,
                    layer_cfg.use_weight,
                )
            if (
                self.attention == "products"
                or self.attention == "reactants_products"
            ):
                self.attention_products = TransConvLayer(
                    layer_cfg.in_channels,
                    layer_cfg.out_channels,
                    layer_cfg.num_heads,
                    layer_cfg.use_weight,
                )
        else:
            if (
                self.attention == "reactants"
                or self.attention == "reactants_products"
            ):
                self.attention_reactants = nn.MultiheadAttention(
                    embed_dim=self.hidden_size, num_heads=layer_cfg.num_heads
                )
            if (
                self.attention == "products"
                or self.attention == "reactants_products"
            ):
                self.attention_products = nn.MultiheadAttention(
                    embed_dim=self.hidden_size, num_heads=layer_cfg.num_heads
                )

    def forward(self, data: object) -> torch.Tensor:
        """
        Forward pass of the GNN.
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch
        atom_origin_type = data.atom_origin_type
        is_real_bond = (
            data.is_real_bond if hasattr(data, "is_real_bond") else None
        )

        # initial edge features
        row, col = edge_index
        h_0 = F.relu(self.edge_init(torch.cat([x[row], edge_attr], dim=1)))
        h = h_0

        # convolutions
        for l in range(self.depth):
            _, h = self.convs[l](edge_index, h, is_real_bond)
            h += h_0
            h = F.dropout(F.relu(h), self.dropout, training=self.training)

        # dmpnn edge -> node aggregation
        s, _ = self.convs[l](
            edge_index, h, is_real_bond
        )  # only use for summing
        q = torch.cat([x, s], dim=1)
        h = F.relu(self.edge_to_node(q))

        new_h = h.clone()

        if self.attention:
            reactant_mask = atom_origin_type == AtomOriginType.REACTANT
            product_mask = atom_origin_type == AtomOriginType.PRODUCT

            updated_reactants, updated_products = self._attention(
                new_h, atom_origin_type
            )

            if self.double_features:
                new_h_double_features = torch.cat(
                    [new_h, torch.zeros_like(new_h)], dim=1
                )
                if (
                    self.attention == "reactants"
                    or self.attention == "reactants_products"
                ):
                    new_h_double_features[
                        reactant_mask, self.hidden_size :
                    ] = updated_reactants
                if (
                    self.attention == "products"
                    or self.attention == "reactants_products"
                ):
                    new_h_double_features[product_mask, self.hidden_size :] = (
                        updated_products
                    )

                new_h = new_h_double_features

            else:
                new_h[reactant_mask] = updated_reactants
                new_h[product_mask] = updated_products

        h = new_h

        # Pooling
        pooled = self._pool(
            h, batch, edge_index, is_real_bond, atom_origin_type
        )

        return self.ffn(pooled).squeeze(-1)

    def _pool(
        self,
        h: torch.Tensor,
        batch: torch.Tensor,
        edge_index: torch.Tensor,
        is_real_bond: Optional[torch.Tensor],
        atom_origin_type: torch.Tensor,
    ) -> torch.Tensor:
        """
        Applies the specified pooling method to the node features.
        """
        if self.pool_type == "global":
            return (
                self._pool_global(h, batch, is_real_bond, edge_index)
                if self.pool_real_only
                else global_add_pool(h, batch)
            )
        elif self.pool_type == "reactants":
            return self._pool_by_type(
                h,
                batch,
                atom_origin_type,
                AtomOriginType.REACTANT,
                is_real_bond,
                edge_index,
            )
        elif self.pool_type == "products":
            return self._pool_by_type(
                h,
                batch,
                atom_origin_type,
                AtomOriginType.PRODUCT,
                is_real_bond,
                edge_index,
            )
        elif self.pool_type == "dummy":
            return self._pool_dummy(h, batch, atom_origin_type)
        else:
            raise ValueError(f"Invalid pool_type: {self.pool_type}")

    def _pool_by_type(
        self,
        h: torch.Tensor,
        batch: torch.Tensor,
        atom_types: torch.Tensor,
        target_type: AtomOriginType,
        is_real_bond: Optional[torch.Tensor],
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pools nodes of a specific type, optionally considering only real bonds.
        """
        if self.pool_real_only:
            return self._pool_real_bonds(
                h, batch, edge_index, is_real_bond, atom_types, target_type
            )
        else:
            mask = atom_types == target_type
            return global_add_pool(h[mask], batch[mask])

    def _pool_global(
        self,
        h: torch.Tensor,
        batch: torch.Tensor,
        is_real_bond: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Performs global pooling, optionally considering only real bonds.
        """
        if self.pool_real_only:
            row, col = edge_index[:, is_real_bond]
            edge_batch = batch[row]
            return scatter_add(h[row], edge_batch, dim=0)
        else:
            return global_add_pool(h, batch)

    def _pool_dummy(
        self, h: torch.Tensor, batch: torch.Tensor, atom_types: torch.Tensor
    ) -> torch.Tensor:
        """
        Pools dummy nodes.
        """
        dummy_mask = atom_types == AtomOriginType.DUMMY
        if not dummy_mask.any():
            raise ValueError("No dummy nodes found in the graph")
        return scatter_add(h[dummy_mask], batch[dummy_mask], dim=0)

    def _pool_real_bonds(
        self,
        h: torch.Tensor,
        batch: torch.Tensor,
        edge_index: torch.Tensor,
        is_real_bond: torch.Tensor,
        atom_types: torch.Tensor,
        target_type: AtomOriginType,
    ) -> torch.Tensor:
        """
        Pools real bonds of a specific atom type.
        """
        row, col = edge_index[:, is_real_bond]
        mask = atom_types[row] == target_type
        row, col = row[mask], col[mask]
        edge_batch = batch[row]
        return scatter_add(h[row], edge_batch, dim=0)

    def _attention(
        self,
        node_features: torch.Tensor,
        atom_origin_type: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies attention between reactants and products.
        If bidirectional_attention is True, applies attention in both directions.
        Otherwise, only updates reactant features based on product features.
        Returns updated features for both reactants and products.
        """
        reactant_mask = atom_origin_type == AtomOriginType.REACTANT
        product_mask = atom_origin_type == AtomOriginType.PRODUCT

        reactant_features = node_features[reactant_mask].unsqueeze(0)
        product_features = node_features[product_mask].unsqueeze(0)

        updated_reactants = reactant_features.clone()
        updated_products = product_features.clone()

        if (
            self.attention == "reactants"
            or self.attention == "reactants_products"
        ):
            if self.fast_attention:
                # Attention: products to reactants
                updated_reactants = self.attention_reactants(
                    reactant_features, product_features
                )
            else:
                # Attention: products to reactants
                updated_reactants, _ = self.attention_reactants(
                    reactant_features,
                    product_features,
                    product_features,
                    need_weights=False,
                )
        if (
            self.attention == "products"
            or self.attention == "reactants_products"
        ):
            if self.fast_attention:
                # Attention: reactants to products
                updated_products = self.attention_products(
                    product_features, reactant_features
                )
            else:
                # Attention: reactants to products
                updated_products, _ = self.attention_products(
                    product_features,
                    reactant_features,
                    reactant_features,
                    need_weights=False,
                )

        return updated_reactants.squeeze(0), updated_products.squeeze(0)
