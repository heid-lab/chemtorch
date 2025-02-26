import hydra
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.utils import to_dense_batch

from deeprxn.layer.mpnn_layer.mpnn_layer_base import MPNNLayerBase


class DeepRXNLayer(MPNNLayerBase):
    def __init__(
        self,
        in_channels,
        out_channels,
        mpnn_cfg,
        att_layer_cfg,
        layer_norm,
        batch_norm,
        dropout,
        log_attn_weights=False,
    ):
        super().__init__(in_channels, out_channels)

        self.local_gnn = hydra.utils.instantiate(mpnn_cfg)
        self.self_attn = hydra.utils.instantiate(att_layer_cfg)
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.log_attn_weights = log_attn_weights

        if self.layer_norm and self.batch_norm:
            raise ValueError(
                "Cannot apply two types of normalization together"
            )

        if self.layer_norm:
            self.norm1_local = pyg_nn.norm.LayerNorm(out_channels)
            self.norm1_attn = pyg_nn.norm.LayerNorm(out_channels)

        if self.batch_norm:
            self.norm1_local = nn.BatchNorm1d(out_channels)
            self.norm1_attn = nn.BatchNorm1d(out_channels)

        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

        # Feed Forward block.
        self.ff_linear1 = nn.Linear(out_channels, out_channels * 2)
        self.ff_linear2 = nn.Linear(out_channels * 2, out_channels)
        self.act_fn_ff = nn.ReLU()

        if self.layer_norm:
            self.norm2 = pyg_nn.norm.LayerNorm(out_channels)

        if self.batch_norm:
            self.norm2 = nn.BatchNorm1d(out_channels)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

    def forward(self, batch):
        h = batch.x
        h_in1 = h

        h_out_list = []
        local_out = self.local_gnn(batch)
        # GatedGCN does residual connection and dropout internally.
        h_local = local_out.x
        batch.edge_attr = local_out.edge_attr

        if self.layer_norm:
            h_local = self.norm1_local(h_local, batch.batch)
        if self.batch_norm:
            h_local = self.norm1_local(h_local)
        h_out_list.append(h_local)

        # h_dense, mask = to_dense_batch(h, batch.batch)
        # h_attn = self._sa_block(h_dense, None, ~mask)[mask]
        # h_attn = self.dropout_attn(h_attn)
        # h_attn = h_in1 + h_attn  # Residual connection.
        # if self.layer_norm:
        #     h_attn = self.norm1_attn(h_attn, batch.batch)
        # if self.batch_norm:
        #     h_attn = self.norm1_attn(h_attn)

        batch = self.self_attn(batch)

        h_out_list.append(batch.h_attn)

        # Combine local and global outputs.
        # h = torch.cat(h_out_list, dim=-1)
        h = sum(h_out_list)

        # Feed Forward block.
        h = h + self._ff_block(h)
        if self.layer_norm:
            h = self.norm2(h, batch.batch)
        if self.batch_norm:
            h = self.norm2(h)

        batch.x = h
        return batch

    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block."""
        if not self.log_attn_weights:
            x = self.self_attn(
                x,
                x,
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )[0]
        else:
            # Requires PyTorch v1.11+ to support `average_attn_weights=False`
            # option to return attention weights of individual heads.
            x, A = self.self_attn(
                x,
                x,
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )
            self.attn_weights = A.detach().cpu()
        return x

    def _ff_block(self, x):
        """Feed Forward block."""
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def extra_repr(self):
        s = (
            f"summary: dim_h={self.dim_h}, "
            f"local_gnn_type={self.local_gnn_type}, "
            f"global_model_type={self.global_model_type}, "
            f"heads={self.num_heads}"
        )
        return s
