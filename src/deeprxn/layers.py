from typing import Optional, Tuple

import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_add


class DMPNNConv(MessagePassing):
    """
    Directed Message Passing Neural Network Convolution layer.
    EXTRA: allowing for separate processing of real and artificial bonds if specified.
    """

    def __init__(
        self,
        hidden_size: int,
        separate_nn: bool = False,
        use_attention_agg: bool = True,
        use_attention_agg_heads: int = 1,
    ):
        super(DMPNNConv, self).__init__(aggr="add")
        self.separate_nn = separate_nn
        self.lin_real = nn.Linear(hidden_size, hidden_size)
        self.lin_artificial = (
            nn.Linear(hidden_size, hidden_size)
            if separate_nn
            else self.lin_real
        )

        self.use_attention_agg = use_attention_agg
        if use_attention_agg:
            self.attention = nn.MultiheadAttention(
                hidden_size,
                num_heads=use_attention_agg_heads,
                dropout=0.02,
                batch_first=True,
            )

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        incoming_edges_list: Optional[torch.Tensor] = None,
        incoming_edges_batch: Optional[torch.Tensor] = None,
        edge_batch: Optional[torch.Tensor] = None,
        is_real_bond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the DMPNNConv layer.
        """
        row, col = edge_index
        max_edges_per_node = 4

        if self.use_attention_agg:
            edge_attr_dense, mask_edge_attr_dense = to_dense_batch(
                edge_attr, batch=edge_batch
            )

            edge_attr_rearranged = edge_attr[
                incoming_edges_list
            ]  # TODO: maybe move to data

            incoming_edges, mask = to_dense_batch(
                edge_attr_rearranged,
                batch=incoming_edges_batch,
                max_num_nodes=max_edges_per_node,
            )

            edge_attr_dense_updated, _ = self.attention(
                edge_attr_dense,
                incoming_edges,
                incoming_edges,
                key_padding_mask=~mask,
                need_weights=False,
            )

            edge_attr_dense_updated = edge_attr_dense_updated[
                mask_edge_attr_dense
            ]

            edge_attr = edge_attr + edge_attr_dense_updated

        aggregated_messages = self.propagate(edge_index, edge_attr=edge_attr)

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


###### from https://github.com/qitianwu/SGFormer/tree/main
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
