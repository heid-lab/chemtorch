from typing import Literal, Optional, Tuple

import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.utils import to_dense_batch

from deeprxn.data import AtomOriginType


class DMPNNConv(MessagePassing):
    """
    Directed Message Passing Neural Network Convolution layer.
    EXTRA: allowing for separate processing of real and artificial bonds if specified.
    """

    def __init__(self, hidden_size: int, separate_nn: bool = False):
        super(DMPNNConv, self).__init__(aggr="add")
        self.separate_nn = separate_nn
        self.lin_real = nn.Linear(hidden_size, hidden_size)
        self.lin_artificial = (
            nn.Linear(hidden_size, hidden_size)
            if separate_nn
            else self.lin_real
        )

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        is_real_bond: Optional[torch.Tensor] = None,
        edge_to_node: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the DMPNNConv layer.
        """
        row, col = edge_index

        aggregated_messages = self.propagate(edge_index, edge_attr=edge_attr)

        if edge_to_node:
            return aggregated_messages, None

        rev_messages = self._compute_reverse_messages(edge_attr)

        if self.separate_nn and is_real_bond is not None:
            out = torch.where(
                is_real_bond.unsqueeze(1),
                self.lin_real(aggregated_messages[row] - rev_messages),
                self.lin_artificial(aggregated_messages[row] - rev_messages),
            )
        else:
            out = self.lin_real(aggregated_messages[row] - rev_messages)

        return aggregated_messages, out

    def message(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Defines the message function for the DMPNNConv layer.
        """
        return edge_attr

    def _compute_reverse_messages(
        self, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the reverse messages for each edge.
        """
        try:
            return torch.flip(
                edge_attr.view(edge_attr.size(0) // 2, 2, -1), dims=[1]
            ).view(edge_attr.size(0), -1)
        except:
            return torch.zeros_like(edge_attr)


class ReactantProductAttention(nn.Module):
    """
    Attention layer for reactants and products in reaction graphs.
    """

    def __init__(
        self,
        hidden_size: int,
        attention: Literal[
            "reactants", "products", "reactants_products"
        ] = "reactants_products",
        num_heads: int = 1,
        dropout: float = 0.02,
        layer_norm: bool = False,
        batch_norm: bool = False,
    ):
        super(ReactantProductAttention, self).__init__()
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

    def forward(
        self,
        node_features: torch.Tensor,
        atom_origin_type: torch.Tensor,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the ReactantProductAttention layer.

        Args:
            node_features: Node features of shape [num_nodes (in whole batch), hidden_size]
            atom_origin_type: Tensor indicating the origin type of each atom [num_nodes (in whole batch)]
            batch: Batch assignment for each node

        Returns:
            Updated node features
        """
        reactant_mask = atom_origin_type == AtomOriginType.REACTANT
        product_mask = atom_origin_type == AtomOriginType.PRODUCT

        # features and batch indices for reactants and products
        reactant_features = node_features[reactant_mask]
        product_features = node_features[product_mask]
        reactant_batch = batch[reactant_mask]
        product_batch = batch[product_mask]

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
        att_output = node_features.clone()

        # residual connection
        if self.attention in ["reactants", "reactants_products"]:
            att_output[reactant_mask] = (
                node_features[reactant_mask] + updated_reactants
            )

        if self.attention in ["products", "reactants_products"]:
            att_output[product_mask] = (
                node_features[product_mask] + updated_products
            )

        # TODO: do we need different layer norm for reactants and products?
        if self.layer_norm:
            att_output = self.ln(att_output, batch)

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


class AttentionMessagePassing(MessagePassing):
    """
    TODO: add docstring
    """

    def __init__(
        self,
        hidden_size: int,
        separate_nn: bool = False,
        num_heads: int = 1,
        dropout: float = 0.02,
        use_message: bool = True,
    ):
        super(AttentionMessagePassing, self).__init__(aggr="add")
        self.separate_nn = separate_nn
        self.use_message = use_message
        self.lin_real = nn.Linear(hidden_size, hidden_size)
        self.lin_artificial = (
            nn.Linear(hidden_size, hidden_size)
            if separate_nn
            else self.lin_real
        )

        self.attention = nn.MultiheadAttention(
            hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        incoming_edges_list: Optional[torch.Tensor] = None,
        edge_batch: Optional[torch.Tensor] = None,
        incoming_edges_batch_from_zero: Optional[torch.Tensor] = None,
        edge_batch_2: Optional[torch.Tensor] = None,
        is_real_bond: Optional[torch.Tensor] = None,
        edge_to_node: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the DMPNNConv layer.
        """

        # used for edge to node agg
        if edge_to_node:
            aggregated_messages = self.propagate(
                edge_index, edge_attr=edge_attr
            )
            return aggregated_messages, None

        row, col = edge_index
        max_edges_per_node = 4  # TODO: automatically

        edge_attr_2 = edge_attr[edge_batch]

        # returns [num_batches, max_nodes_per_batch, hidden_size], nodes here are edges
        # here there is one edge in a batch
        edge_attr_dense, mask_edge_attr_dense = to_dense_batch(
            edge_attr_2, batch=edge_batch_2
        )

        # get attrs for incoming edges
        edge_attr_rearranged = edge_attr[
            incoming_edges_list
        ]  # TODO: maybe move to data

        # to_dense_batch (nodes are edges here)
        # [num_batches, max_nodes_per_batch, hidden_size], [num_batches, max_nodes_per_batch]
        # the mask indicates for each batch which nodes are real and which are padding
        incoming_edges, mask = to_dense_batch(
            edge_attr_rearranged,
            batch=incoming_edges_batch_from_zero,
            max_num_nodes=max_edges_per_node,
        )

        # Q: single edges, K: respective incoming edges, V: respective incoming edges
        edge_attr_dense_updated, _ = self.attention(
            edge_attr_dense,
            incoming_edges,
            incoming_edges,
            key_padding_mask=~mask,
            need_weights=False,
        )

        # unmask
        edge_attr_dense_updated = edge_attr_dense_updated[mask_edge_attr_dense]

        # residual connection
        edge_attr_att_output = edge_attr.clone()
        edge_attr_att_output[edge_batch] += edge_attr_dense_updated

        if not self.use_message:
            return None, edge_attr_att_output

        # TODO: further investigate
        # Including the message passing mechanism does improve results
        aggregated_messages = self.propagate(
            edge_index, edge_attr=edge_attr_att_output
        )

        rev_messages = self._compute_reverse_messages(edge_attr_att_output)

        if self.separate_nn and is_real_bond is not None:
            out = torch.where(
                is_real_bond.unsqueeze(1),
                self.lin_real(aggregated_messages[row] - rev_messages),
                self.lin_artificial(aggregated_messages[row] - rev_messages),
            )
        else:
            out = self.lin_real(aggregated_messages[row] - rev_messages)

        return aggregated_messages, out

    def message(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Defines the message function for the DMPNNConv layer.
        """
        return edge_attr

    def _compute_reverse_messages(
        self, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the reverse messages for each edge.
        """
        try:
            return torch.flip(
                edge_attr.view(edge_attr.size(0) // 2, 2, -1), dims=[1]
            ).view(edge_attr.size(0), -1)
        except:
            return torch.zeros_like(edge_attr)


class AttentionMessagePassingTransEnc(MessagePassing):
    """
    TODO: add docstring
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 1,
        dropout: float = 0.02,
        layer_norm: bool = False,
        batch_norm: bool = False,
    ):
        super(AttentionMessagePassingTransEnc, self).__init__()
        self.hidden_size = hidden_size
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        self.attention = nn.MultiheadAttention(
            hidden_size,
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

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        incoming_edges_list: Optional[torch.Tensor] = None,
        incoming_edges_batch: Optional[torch.Tensor] = None,
        edge_batch: Optional[torch.Tensor] = None,
        is_real_bond: Optional[torch.Tensor] = None,
        edge_to_node: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the DMPNNConv layer.
        """

        # used for edge to node agg
        if edge_to_node:
            aggregated_messages = self.propagate(
                edge_index, edge_attr=edge_attr
            )
            return aggregated_messages, None

        row, col = edge_index
        max_edges_per_node = 4

        # returns [num_batches, max_nodes_per_batch, hidden_size], nodes here are edges
        # here there is one edge in a batch
        edge_attr_dense, mask_edge_attr_dense = to_dense_batch(
            edge_attr, batch=edge_batch
        )

        # get attrs for incoming edges
        edge_attr_rearranged = edge_attr[
            incoming_edges_list
        ]  # TODO: maybe move to data

        # to_dense_batch (nodes are edges here)
        # [num_batches, max_nodes_per_batch, hidden_size], [num_batches, max_nodes_per_batch]
        # the mask indicates for each batch which nodes are real and which are padding
        incoming_edges, mask = to_dense_batch(
            edge_attr_rearranged,
            batch=incoming_edges_batch,
            max_num_nodes=max_edges_per_node,
        )

        # Q: single edges, K: respective incoming edges, V: respective incoming edges
        edge_attr_dense_updated, _ = self.attention(
            edge_attr_dense,
            incoming_edges,
            incoming_edges,
            key_padding_mask=~mask,
            need_weights=False,
        )

        # unmask
        edge_attr_dense_updated = edge_attr_dense_updated[mask_edge_attr_dense]

        # residual connection
        att_output = edge_attr + edge_attr_dense_updated

        if self.layer_norm:
            att_output = self.ln(att_output, edge_batch)

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

        return None, att_output

    def message(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Defines the message function for the DMPNNConv layer.
        """
        return edge_attr


###### from https://github.com/qitianwu/SGFormer/tree/main
###### IGNORE FOR NOW, outdated, TODO: further adapt to new changes
class TransConvLayer(nn.Module):
    """
    transformer with fast attention
    """

    def __init__(self, in_channels, out_channels, num_heads, use_weight=True):
        super().__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()

    def forward(
        self,
        query_input,
        source_input,
        edge_index=None,
        edge_weight=None,
        output_attn=False,
    ):
        # feature transformation
        query = self.Wq(query_input).reshape(
            -1, self.num_heads, self.out_channels
        )
        key = self.Wk(source_input).reshape(
            -1, self.num_heads, self.out_channels
        )
        if self.use_weight:
            value = self.Wv(source_input).reshape(
                -1, self.num_heads, self.out_channels
            )
        else:
            value = source_input.reshape(-1, 1, self.out_channels)

        # compute full attentive aggregation
        if output_attn:
            attention_output, attn = full_attention_conv(
                query, key, value, output_attn
            )  # [N, H, D]
        else:
            attention_output = full_attention_conv(
                query, key, value
            )  # [N, H, D]

        final_output = attention_output
        final_output = final_output.mean(dim=1)

        if output_attn:
            return final_output, attn
        else:
            return final_output


def full_attention_conv(qs, ks, vs, output_attn=False):
    # normalize input
    qs = qs / torch.norm(qs, p=2)  # [N, H, M]
    ks = ks / torch.norm(ks, p=2)  # [L, H, M]
    N = qs.shape[0]

    # numerator
    kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
    attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)  # [N, H, D]
    attention_num += N * vs

    # denominator
    all_ones = torch.ones([ks.shape[0]]).to(ks.device)
    ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
    attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

    # attentive aggregated results
    attention_normalizer = torch.unsqueeze(
        attention_normalizer, len(attention_normalizer.shape)
    )  # [N, H, 1]
    attention_normalizer += torch.ones_like(attention_normalizer) * N
    attn_output = attention_num / attention_normalizer  # [N, H, D]

    # compute attention for visualization if needed
    if output_attn:
        attention = torch.einsum("nhm,lhm->nlh", qs, ks).mean(dim=-1)  # [N, N]
        normalizer = attention_normalizer.squeeze(dim=-1).mean(
            dim=-1, keepdims=True
        )  # [N,1]
        attention = attention / normalizer

    if output_attn:
        return attn_output, attention
    else:
        return attn_output
