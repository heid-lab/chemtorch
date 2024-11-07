from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn
from torch_geometric.nn import GATConv, global_add_pool
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_add

from deeprxn.mpnn_layer.layers import (
    AttentionMessagePassing,
    AttentionMessagePassingTransEnc,
    DMPNNConv,
    ReactantProductAttention,
)
from deeprxn.representation.data import AtomOriginType

# TODO: look into fast attention again


class GNN(nn.Module):
    """
    TODO: Add docstring
    """

    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        depth: int,
        hidden_size: int,
        dropout: float,
        attention: Literal[
            None, "reactants", "products", "reactants_products"
        ],
        pool_type: Literal["global", "reactants", "products", "dummy"],  # DONE
        pool_real_only: bool,  # TODO: look into this
        double_features: bool,
        use_attention_agg: bool,
        use_attention_node_update: bool,
        use_attention_node_update_heads: int,
        attention_depth: int,
        shared_weights: bool,
        use_att_agg_trans_enc: bool,
        return_attention_weights: bool,
        dmpnn_conv_cfg: DictConfig,
        react_prod_att_cfg: DictConfig,
        att_mess_pass_cfg: DictConfig,
        att_mess_pass_trans_enc_cfg: DictConfig,
    ):
        super(GNN, self).__init__()
        self.depth = depth
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.pool_type = pool_type
        self.pool_real_only = pool_real_only
        self.double_features = double_features and attention is not None
        self.attention = attention
        self.use_attention_agg = use_attention_agg
        self.use_attention_node_update = use_attention_node_update
        self.use_attention_node_update_heads = use_attention_node_update_heads
        self.attention_depth = attention_depth
        self.use_att_agg_trans_enc = use_att_agg_trans_enc
        self.return_attention_weights = return_attention_weights

        self.edge_init = nn.Linear(
            num_node_features + num_edge_features, self.hidden_size
        )

        if shared_weights:
            if self.use_attention_agg:
                if self.use_att_agg_trans_enc:
                    self.conv = AttentionMessagePassingTransEnc(
                        hidden_size=self.hidden_size,
                        num_heads=att_mess_pass_trans_enc_cfg.num_heads,
                        dropout=att_mess_pass_trans_enc_cfg.dropout,
                        layer_norm=att_mess_pass_trans_enc_cfg.layer_norm,
                        batch_norm=att_mess_pass_trans_enc_cfg.batch_norm,
                    )
                else:
                    self.conv = AttentionMessageAggregation(
                        hidden_size=self.hidden_size,
                        separate_nn=att_mess_pass_cfg.separate_nn,
                        num_heads=att_mess_pass_cfg.num_heads,
                        dropout=att_mess_pass_cfg.dropout,
                        use_message=att_mess_pass_cfg.use_message,
                    )
            else:
                self.conv = DMPNNConv(
                    self.hidden_size, dmpnn_conv_cfg.separate_nn
                )

        self.convs = torch.nn.ModuleList()
        for _ in range(self.depth):
            if self.use_attention_agg:
                if shared_weights:
                    self.convs.append(self.conv)
                else:
                    if self.use_att_agg_trans_enc:
                        self.convs.append(
                            AttentionMessagePassingTransEnc(
                                hidden_size=self.hidden_size,
                                num_heads=att_mess_pass_trans_enc_cfg.num_heads,
                                dropout=att_mess_pass_trans_enc_cfg.dropout,
                                layer_norm=att_mess_pass_trans_enc_cfg.layer_norm,
                                batch_norm=att_mess_pass_trans_enc_cfg.batch_norm,
                            )
                        )
                    else:
                        self.convs.append(
                            AttentionMessageAggregation(
                                hidden_size=self.hidden_size,
                                separate_nn=att_mess_pass_cfg.separate_nn,
                                num_heads=att_mess_pass_cfg.num_heads,
                                dropout=att_mess_pass_cfg.dropout,
                                use_message=att_mess_pass_cfg.use_message,
                            )
                        )
            else:
                if shared_weights:
                    self.convs.append(self.conv)
                else:
                    self.convs.append(
                        DMPNNConv(self.hidden_size, dmpnn_conv_cfg.separate_nn)
                    )

        self.edge_to_node = nn.Linear(
            num_node_features + self.hidden_size, self.hidden_size
        )

        if self.use_attention_node_update:
            self.node_update_attention = nn.MultiheadAttention(
                embed_dim=self.hidden_size,
                num_heads=use_attention_node_update_heads,
                dropout=0.02,
                batch_first=True,
            )
            self.node_linear = nn.Linear(num_node_features, self.hidden_size)

        if self.attention:
            if shared_weights:
                self.att_layer = ReactantProductAttention(
                    hidden_size=self.hidden_size,
                    attention=self.attention,
                    num_heads=react_prod_att_cfg.num_heads,
                    dropout=react_prod_att_cfg.dropout,
                    layer_norm=react_prod_att_cfg.layer_norm,
                    batch_norm=react_prod_att_cfg.batch_norm,
                )
            self.att_layers = torch.nn.ModuleList()

            for _ in range(self.attention_depth):
                if shared_weights:
                    self.att_layers.append(self.att_layer)
                else:
                    self.att_layers.append(
                        ReactantProductAttention(
                            hidden_size=self.hidden_size,
                            attention=self.attention,
                            num_heads=react_prod_att_cfg.num_heads,
                            dropout=react_prod_att_cfg.dropout,
                            layer_norm=react_prod_att_cfg.layer_norm,
                            batch_norm=react_prod_att_cfg.batch_norm,
                        )
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

    def forward(self, data: object) -> torch.Tensor:
        """
        Forward pass of the GNN.
        """
        # TODO: clean up data extraction and improve efficiency
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch
        atom_origin_type = data.atom_origin_type
        if self.use_attention_agg:
            incoming_edges_list = data.incoming_edges_list
            incoming_edges_batch = data.incoming_edges_batch
            edge_batch = torch.unique(incoming_edges_batch, sorted=False)
            incoming_edges_batch_from_zero = (
                data.incoming_edges_batch_from_zero
            )
            edge_batch_2 = torch.arange(
                edge_batch.size(0), device=edge_batch.device
            )
        else:
            incoming_edges_list = None
            incoming_edges_batch = None
            edge_batch = None

        # TODO: check if incoming edges lists are same
        if self.use_attention_node_update:
            node_batch = torch.arange(x.size(0), device=x.device)
            neighboring_nodes_list = data.neighboring_nodes_list
            neighboring_nodes_batch = data.neighboring_nodes_batch
            incoming_edges_nodes_list = data.incoming_edges_nodes_list
            incoming_edges_nodes_batch = data.incoming_edges_nodes_batch
        else:
            node_batch = None
            neighboring_nodes_list = None
            neighboring_nodes_batch = None
            incoming_edges_nodes_list = None
            incoming_edges_nodes_batch = None

        is_real_bond = (
            data.is_real_bond if hasattr(data, "is_real_bond") else None
        )

        # initial edge features
        row, col = edge_index
        h_0 = F.relu(self.edge_init(torch.cat([x[row], edge_attr], dim=1)))
        h = h_0

        # convolutions
        # TODO: write without if statement
        if not self.use_attention_agg:
            # normal dmpnn
            for l in range(self.depth):
                _, h = self.convs[l](
                    edge_index=edge_index,
                    edge_attr=h,
                    is_real_bond=is_real_bond,
                )
                h += h_0
                h = F.dropout(F.relu(h), self.dropout, training=self.training)
        else:
            for l in range(self.depth):
                # convolutions using attention
                _, h = self.convs[l](
                    edge_index=edge_index,
                    edge_attr=h,
                    incoming_edges_list=incoming_edges_list,
                    incoming_edges_batch=incoming_edges_batch,
                    edge_batch=edge_batch,
                    incoming_edges_batch_from_zero=incoming_edges_batch_from_zero,
                    edge_batch_2=edge_batch_2,
                    is_real_bond=is_real_bond,
                )
                if not self.use_att_agg_trans_enc:
                    h += h_0
                    h = F.dropout(
                        F.relu(h), self.dropout, training=self.training
                    )

        if not self.use_attention_node_update:
            # dmpnn edge -> node aggregation
            s, _ = self.convs[l](
                edge_index,
                h,
                is_real_bond,
                debug=False,
                # lol=False,
                # edge_batch=batch[row],
                # return_attention_weights=self.return_attention_weights,
            )  # only use for summing
        else:
            x_h_dim = self.node_linear(x)

            # dmpnn edge -> node aggregation, using attention
            s = self._edge_to_node_agg(
                x=x_h_dim,
                node_batch=node_batch,
                neighboring_nodes_list=neighboring_nodes_list,
                neighboring_nodes_batch=neighboring_nodes_batch,
                h=h,
                incoming_edges_nodes_list=incoming_edges_nodes_list,
                incoming_edges_nodes_batch=incoming_edges_nodes_batch,
            )

        q = torch.cat([x, s], dim=1)
        h = F.relu(self.edge_to_node(q))

        # attention layer
        if self.attention:
            for l in range(self.attention_depth):
                h = self.att_layers[l](
                    h,
                    atom_origin_type,
                    batch,
                )
        # print(h)

        # [num_nodes, hidden_size] -> [num_batches, hidden_size] (hidden might be *2 if double feat)
        pooled = self._pool(
            h, batch, edge_index, is_real_bond, atom_origin_type
        )

        # print(
        #     "pooled",
        #     torch.isnan(pooled).any(),
        # )
        # print(pooled)

        return self.ffn(pooled).squeeze(-1)

    # TODO: look into what exactly happens when there are no incoming edges, specifically data.py
    def _edge_to_node_agg(
        self,
        x: torch.Tensor,
        node_batch: torch.Tensor,
        neighboring_nodes_list: torch.Tensor,
        neighboring_nodes_batch: torch.Tensor,
        h: torch.Tensor,
        incoming_edges_nodes_list: torch.Tensor,
        incoming_edges_nodes_batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        TODO: add doc
        """

        # returns [num_batches, max_nodes_per_batch, hidden_size]
        # here there is one node in a batch
        node_dense, node_mask = to_dense_batch(x, batch=node_batch)

        # get attrs for neighboring nodes
        neighboring_x = x[neighboring_nodes_list]

        # to_dense_batch
        # [num_batches, max_nodes_per_batch, hidden_size], [num_batches, max_nodes_per_batch]
        # the mask indicates for each batch which nodes are real and which are padding
        neighboring_x_dense, neighboring_x_mask = to_dense_batch(
            neighboring_x,
            batch=neighboring_nodes_batch,
            max_num_nodes=6,
        )

        # get attrs for incoming edges
        incoming_h = h[incoming_edges_nodes_list]

        # [num_batches, max_nodes_per_batch, hidden_size], here nodes are edges
        incoming_h_dense, incoming_h_mask = to_dense_batch(
            incoming_h,
            batch=incoming_edges_nodes_batch,
            max_num_nodes=6,
        )

        # Q: single nodes, K: respective neighboring nodes, V: respective incoming edges
        node_dense_updated, _ = self.node_update_attention(
            node_dense,
            neighboring_x_dense,
            incoming_h_dense,
            key_padding_mask=~neighboring_x_mask,
            need_weights=False,
        )

        # unmask
        s = node_dense_updated[node_mask]

        # residual connection
        s = x + s

        return s

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


class GAT(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        depth: int,
        hidden_size: int,
        dropout: float,
        attention: Literal[
            None, "reactants", "products", "reactants_products"
        ],
        pool_type: Literal["global", "reactants", "products", "dummy"],
        double_features: bool,
        shared_weights: bool,
        attention_depth: int,
        gat_cfg: DictConfig,
        react_prod_att_cfg: DictConfig,
    ):
        super(GAT, self).__init__()
        self.depth = depth
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.pool_type = pool_type
        self.double_features = double_features and attention is not None
        self.attention = attention
        self.attention_depth = attention_depth
        self.skip_drop = gat_cfg.skip_drop
        self.pool_real_only = False

        # self.edge_init = nn.Linear(
        #     num_node_features + num_edge_features, self.hidden_size
        # )
        self.edge_init = nn.Linear(num_node_features, self.hidden_size)

        if shared_weights:
            self.conv = GATConv(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size // gat_cfg.num_heads,
                heads=gat_cfg.num_heads,
                dropout=gat_cfg.dropout,
                edge_dim=num_edge_features,
            )

        self.convs = torch.nn.ModuleList()
        for _ in range(self.depth):
            if shared_weights:
                self.convs.append(self.conv)
            else:
                self.convs.append(
                    GATConv(
                        in_channels=self.hidden_size,
                        out_channels=self.hidden_size // gat_cfg.num_heads,
                        heads=gat_cfg.num_heads,
                        dropout=gat_cfg.dropout,
                        edge_dim=num_edge_features,
                    )
                )

        # self.edge_to_node = nn.Linear(
        #     num_node_features + self.hidden_size, self.hidden_size
        # )

        if self.attention:
            if shared_weights:
                self.att_layer = ReactantProductAttention(
                    hidden_size=self.hidden_size,
                    attention=self.attention,
                    num_heads=react_prod_att_cfg.num_heads,
                    dropout=react_prod_att_cfg.dropout,
                    layer_norm=react_prod_att_cfg.layer_norm,
                    batch_norm=react_prod_att_cfg.batch_norm,
                )
            self.att_layers = torch.nn.ModuleList()

            for _ in range(self.attention_depth):
                if shared_weights:
                    self.att_layers.append(self.att_layer)
                else:
                    self.att_layers.append(
                        ReactantProductAttention(
                            hidden_size=self.hidden_size,
                            attention=self.attention,
                            num_heads=react_prod_att_cfg.num_heads,
                            dropout=react_prod_att_cfg.dropout,
                            layer_norm=react_prod_att_cfg.layer_norm,
                            batch_norm=react_prod_att_cfg.batch_norm,
                        )
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

    def forward(self, data: object) -> torch.Tensor:
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
        # h_0 = F.relu(self.edge_init(torch.cat([x[row], edge_attr], dim=1)))
        h_0 = F.relu(self.edge_init(x))
        h = h_0

        # print("x", x.shape)
        # print("edge_index", edge_index.shape)
        # print("edge_attr", edge_attr.shape)

        for l in range(self.depth):
            h = self.convs[l](
                x=h,
                edge_index=edge_index,
                edge_attr=edge_attr,
            )
            if self.skip_drop:
                h += h_0
                h = F.dropout(F.relu(h), self.dropout, training=self.training)

        # q = torch.cat([x, s], dim=1)
        # h = F.relu(self.edge_to_node(q))

        # attention layer
        if self.attention:
            for l in range(self.attention_depth):
                h = self.att_layers[l](
                    h,
                    atom_origin_type,
                    batch,
                )

        # [num_nodes, hidden_size] -> [num_batches, hidden_size] (hidden might be *2 if double feat)
        pooled = self._pool(
            h, batch, edge_index, is_real_bond, atom_origin_type
        )

        # print(
        #     "pooled",
        #     torch.isnan(pooled).any(),
        # )

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
