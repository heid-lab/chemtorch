import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
from omegaconf import DictConfig
from torch_geometric.data import Batch

from deeprxn.model.model_base import Model
from deeprxn.representation.reaction_graph import AtomOriginType


class MaskedDMG(Model):
    """Custom model using configurable components."""

    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        hidden_channels: int,
        layers,
        separate: bool,
        encoder_cfg: DictConfig,
        layer_cfg: DictConfig,
        pool_cfg: DictConfig,
        head_cfg: DictConfig,
    ):
        """Initialize Custom model."""
        super().__init__()
        self.separate = separate

        self.encoders = nn.ModuleList()
        for _, config in encoder_cfg.items():
            self.encoders.append(hydra.utils.instantiate(config))

        self.layers = nn.ModuleList()
        for layer in layers:
            partial_layer = hydra.utils.instantiate(layer_cfg)
            self.layers.append(partial_layer(mode=layer))

        self.pool = hydra.utils.instantiate(pool_cfg)
        self.head = hydra.utils.instantiate(head_cfg)

    def forward(self, batch: Batch) -> Batch:
        """Forward pass through Custom model."""

        for encoder in self.encoders:
            batch = encoder(batch)

        for layer in self.layers:
            batch = layer(batch)

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
            batch_reactant_x = self.pool(batch_reactant)
            batch_product_x = self.pool(batch_product)
            batch.x = torch.cat([batch_reactant_x, batch_product_x], dim=-1)
        else:
            batch.x = self.pool(batch)

        preds = self.head(batch)

        return preds
