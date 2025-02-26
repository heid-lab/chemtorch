import hydra
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.nn.aggr import SumAggregation
from torch_geometric.utils import to_dense_batch

from deeprxn.layer.mpnn_layer.mpnn_layer_base import MPNNLayerBase
from deeprxn.representation.rxn_graph_base import AtomOriginType


class DMPNNDeepRXNLayer20(MPNNLayerBase):
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
        self.self_attn_reactants = hydra.utils.instantiate(att_layer_cfg)
        self.self_attn_products = hydra.utils.instantiate(att_layer_cfg)
        self.self_attn_global = hydra.utils.instantiate(att_layer_cfg)
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.log_attn_weights = log_attn_weights

        if self.layer_norm and self.batch_norm:
            raise ValueError(
                "Cannot apply two types of normalization together"
            )

        if self.layer_norm:
            self.norm1_local = pyg_nn.norm.LayerNorm(out_channels)
            self.norm1_attn_1 = pyg_nn.norm.LayerNorm(out_channels)
            self.norm1_attn_2 = pyg_nn.norm.LayerNorm(out_channels)

        if self.batch_norm:
            self.norm1_local = nn.BatchNorm1d(out_channels)
            self.norm1_attn_1 = nn.BatchNorm1d(out_channels)
            self.norm1_attn_2 = nn.BatchNorm1d(out_channels)

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
        self.aggregation = SumAggregation()

    def forward(self, batch):
        h = batch.x
        # batch.h_0 = batch.edge_attr
        h_in1 = h

        h_out_list = []
        local_out = self.local_gnn(batch)
        s = self.aggregation(local_out.h, local_out.edge_index[1])
        h_local = local_out.x + s
        # batch.edge_attr = local_out.edge_attr

        if self.layer_norm:
            h_local = self.norm1_local(h_local, batch.batch)
        if self.batch_norm:
            h_local = self.norm1_local(h_local)
        h_out_list.append(h_local)

        reactant_mask = batch.atom_origin_type == AtomOriginType.REACTANT
        product_mask = batch.atom_origin_type == AtomOriginType.PRODUCT
        reactant_features = h[reactant_mask]
        product_features = h[product_mask]
        reactant_batch = batch.batch[reactant_mask]
        product_batch = batch.batch[product_mask]
        reactant_features_dense, reactant_mask_dense = to_dense_batch(
            reactant_features, reactant_batch
        )
        product_features_dense, product_mask_dense = to_dense_batch(
            product_features, product_batch
        )
        h_dense, mask = to_dense_batch(h, batch.batch)
        h_attn_reactants = self._sa_reactants_block(
            reactant_features_dense, None, ~reactant_mask_dense
        )[reactant_mask_dense]
        h_attn_products = self._sa_products_block(
            product_features_dense, None, ~product_mask_dense
        )[reactant_mask_dense]
        h_attn_global = self._sa_global_block(h_dense, None, ~mask)[mask]

        h_attn_reactants = self.dropout_attn(h_attn_reactants)
        h_attn_products = self.dropout_attn(h_attn_products)
        h_attn_global = self.dropout_attn(h_attn_global)

        h_attn_reactants = h[reactant_mask] + h_attn_reactants
        h_attn_products = h[product_mask] + h_attn_products

        combined_features = h.clone()
        combined_features[reactant_mask] = h_attn_reactants
        combined_features[product_mask] = h_attn_products

        h_attn = h_in1 + h_attn_global

        if self.layer_norm:
            h_attn = self.norm1_attn_1(h_attn, batch.batch)
            combined_features = self.norm1_attn_2(
                combined_features, batch.batch
            )
        if self.batch_norm:
            h_attn = self.norm1_attn_1(h_attn)
            combined_features = self.norm1_attn_2(combined_features)
        h_out_list.append(h_attn)
        h_out_list.append(combined_features)

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

    def _sa_reactants_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block."""
        if not self.log_attn_weights:
            x = self.self_attn_reactants(
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
            x, A = self.self_attn_reactants(
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

    def _sa_products_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block."""
        if not self.log_attn_weights:
            x = self.self_attn_products(
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
            x, A = self.self_attn_products(
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

    def _sa_global_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block."""
        if not self.log_attn_weights:
            x = self.self_attn_global(
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
            x, A = self.self_attn_global(
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
