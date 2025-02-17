import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
from omegaconf import DictConfig
from torch_geometric.data import Batch
from torch_geometric.nn.aggr import SumAggregation

from deeprxn.model.model_base import Model
from deeprxn.representation.rxn_graph_base import AtomOriginType


class Masked(Model):
    """Custom model using configurable components."""

    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        hidden_channels: int,
        separate: bool,
        separate_pool: bool,
        dmpnn_depth: int,
        layers,
        shared_weights: bool,
        residual: bool,
        encoder_cfg: DictConfig,
        layer_cfg: DictConfig,
        pool_cfg: DictConfig,
        head_cfg: DictConfig,
        dmpnn_layer_cfg=None,
        dataset_precomputed=None,
    ):
        """Initialize Custom model."""
        super().__init__()
        self.dmpnn_depth = dmpnn_depth
        self.separate = separate
        self.separate_pool = separate_pool
        self.residual = residual
        rwpe_enc_out = 0
        deg_enc_out = 0

        self.encoders = nn.ModuleList()
        for _, config in encoder_cfg.items():
            self.encoders.append(hydra.utils.instantiate(config))
            if (
                config._target_ == "deeprxn.encoder.rwpe_enc.RWEncoder"
            ):  # TODO generalize
                rwpe_enc_out = config.out_channels
            if (
                config._target_ == "deeprxn.encoder.deg_enc.DegreeEncoder"
            ):  # TODO generalize
                deg_enc_out = config.out_channels

        if dmpnn_layer_cfg is not None:
            self.dmpnn_layers = nn.ModuleList()
            for _ in range(self.dmpnn_depth):
                self.dmpnn_layers.append(
                    hydra.utils.instantiate(dmpnn_layer_cfg)
                )
            self.aggregation = SumAggregation()
            self.edge_to_node = nn.Linear(
                num_node_features
                + hidden_channels
                + rwpe_enc_out
                + deg_enc_out,
                hidden_channels,
            )

        self.layers = nn.ModuleList()
        for layer in layers:
            partial_layer = hydra.utils.instantiate(layer_cfg)
            self.layers.append(partial_layer(mode=layer))

        if self.separate:
            if self.separate_pool:
                self.pool_reactant = hydra.utils.instantiate(pool_cfg)
                self.pool_product = hydra.utils.instantiate(pool_cfg)
            else:
                self.pool = hydra.utils.instantiate(pool_cfg)
        else:
            self.pool = hydra.utils.instantiate(pool_cfg)

        self.head = hydra.utils.instantiate(head_cfg)

    def forward(self, batch: Batch) -> Batch:
        """Forward pass through Custom model."""

        for encoder in self.encoders:
            batch = encoder(batch)

        if hasattr(self, "dmpnn_layers"):
            for dmpnn_layer in self.dmpnn_layers:
                batch = dmpnn_layer(batch)
            s = self.aggregation(batch.h, batch.edge_index[1])
            batch.q = torch.cat([batch.x, s], dim=1)
            batch.x = F.relu(self.edge_to_node(batch.q))

        for layer in self.layers:
            if self.residual:
                pre_layer = batch.x
            batch = layer(batch)
            if self.residual:
                batch.x = pre_layer + batch.x

        if self.separate:
            num_graphs = batch.ptr.size(0) - 1
            reactant_graphs = []
            product_graphs = []
            for graph_idx in range(num_graphs):
                node_mask = batch.batch == graph_idx
                edge_mask = batch.batch[batch.edge_index[0]] == graph_idx
                node_reactant_mask = node_mask & (
                    batch.atom_origin_type == AtomOriginType.REACTANT
                )
                reactant = tg.data.Data()
                reactant.x = batch.x[node_reactant_mask]
                reactant_graphs.append(reactant)
                node_product_mask = node_mask & (
                    batch.atom_origin_type == AtomOriginType.PRODUCT
                )
                product = tg.data.Data()
                product.x = batch.x[node_product_mask]
                product_graphs.append(product)

            batch_reactant = tg.data.Batch.from_data_list(reactant_graphs)
            batch_product = tg.data.Batch.from_data_list(product_graphs)
            if self.separate_pool:
                batch_reactant_x = self.pool_reactant(batch_reactant)
                batch_product_x = self.pool_product(batch_product)
            else:
                batch_reactant_x = self.pool(batch_reactant)
                batch_product_x = self.pool(batch_product)
            batch.x = torch.cat([batch_reactant_x, batch_product_x], dim=-1)
        else:
            batch.x = self.pool(batch)

        preds = self.head(batch)

        return preds
