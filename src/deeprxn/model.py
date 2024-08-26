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
        pool_type: Literal[
            "global", "reactants", "products", "dummy"
        ] = "global",
        pool_real_only: bool = False,  # TODO: look into this
        react_feat_concat: bool = False,
    ):
        super(GNN, self).__init__()
        self.depth = depth
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.pool_type = pool_type
        self.separate_nn = layer_cfg.separate_nn
        self.pool_real_only = pool_real_only
        self.react_feat_concat = react_feat_concat

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
            self.hidden_size * 2
            if self.react_feat_concat
            else self.hidden_size
        )
        self.ffn = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(ffn_input_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 1),
        )

        self.use_attention = layer_cfg.use_attention
        if self.use_attention:
            # self.attention = nn.MultiheadAttention(hidden_size, num_heads)
            self.attention = TransConvLayer(
                layer_cfg.in_channels,
                layer_cfg.out_channels,
                layer_cfg.num_heads,
                layer_cfg.use_weight,
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

        if self.use_attention and self.pool_type in ["global", "reactants"]:
            if self.react_feat_concat:
                original_h, updated_h = self._update_reactants_with_attention(
                    h, batch, atom_origin_type
                )
                h = torch.cat([original_h, updated_h], dim=1)
            else:
                h = self._update_reactants_with_attention(
                    h, batch, atom_origin_type
                )

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
            if self.react_feat_concat:
                return global_add_pool(h, batch[mask])
            else:
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

    def _update_reactants_with_attention(
        self,
        node_features: torch.Tensor,
        batch: torch.Tensor,
        atom_origin_type: torch.Tensor,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates reactant features using attention mechanism.
        If react_feat_concat is True, returns both original and updated features.
        """
        reactant_mask = atom_origin_type == AtomOriginType.REACTANT
        product_mask = atom_origin_type == AtomOriginType.PRODUCT

        reactant_features = node_features[reactant_mask].unsqueeze(0)
        product_features = node_features[product_mask].unsqueeze(0)

        attn_output = self.attention(reactant_features, product_features)

        if self.react_feat_concat:
            original_reactant_features = node_features[reactant_mask]
            return original_reactant_features, attn_output.squeeze(0)
        else:
            updated_node_features = node_features.clone()
            updated_node_features[reactant_mask] = attn_output.squeeze(0)
            return updated_node_features
