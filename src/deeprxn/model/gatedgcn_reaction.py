import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
from omegaconf import DictConfig
from torch_geometric.data import Batch
from torch_geometric.nn.aggr import SumAggregation
from torch_geometric.utils import to_dense_batch

from deeprxn.model.model_base import Model
from deeprxn.representation.reaction_graph import AtomOriginType


class GatedGCNReaction(Model):
    """Custom model using configurable components."""

    def __init__(
        self,
        num_node_features,
        num_edge_features,
        layer_cfg,
        hidden_channels,
        separate: bool,
        shared_weights: bool,
        mpnn_depth: int,
        encoder_cfg: DictConfig,
        mpnn_layer_cfg: DictConfig,
        attention_layer_cfg: DictConfig,
        pool_cfg: DictConfig,
        head_cfg: DictConfig,
        dataset_precomputed=None,
    ):
        """Initialize Custom model."""
        super().__init__()
        self.mpnn_depth = mpnn_depth
        self.separate = separate

        self.encoders = nn.ModuleList()
        for _, config in encoder_cfg.items():
            self.encoders.append(hydra.utils.instantiate(config))

        self.mpnn_layers = nn.ModuleList()
        if shared_weights:
            layer = hydra.utils.instantiate(
                mpnn_layer_cfg
            )
            for _ in range(self.mpnn_depth):
                self.mpnn_layers.append(layer)
        else:
            for _ in range(self.mpnn_depth):
                self.mpnn_layers.append(
                    hydra.utils.instantiate(
                        mpnn_layer_cfg
                    )
                )

        self.attention_layer = hydra.utils.instantiate(attention_layer_cfg)

        self.pool = hydra.utils.instantiate(pool_cfg)

        self.head = hydra.utils.instantiate(head_cfg)

    def forward(self, batch: Batch) -> Batch:
        """Forward pass through Custom model."""

        for encoder in self.encoders:
            batch = encoder(batch)

        for mpnn_layer in self.mpnn_layers:
            batch = mpnn_layer(batch)

        # this is just one layer for now
        batch = self.attention_layer(batch)

        reactant_mask = batch.atom_origin_type == AtomOriginType.REACTANT
        product_mask = batch.atom_origin_type == AtomOriginType.PRODUCT
        reactant_features = batch.x[reactant_mask]
        product_features = batch.x[product_mask]
        reactant_batch_indices = batch.batch[reactant_mask]
        product_batch_indices = batch.batch[product_mask]

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

            # i think for just sum pool, there is SumAggregation()(reactant_features, reactant_batch_indices, dim_size=batch.num_graphs)

            batch_reactant = tg.data.Batch.from_data_list(reactant_graphs)
            batch_product = tg.data.Batch.from_data_list(product_graphs)
            batch_reactant_x = self.pool(batch_reactant)
            batch_product_x = self.pool(batch_product)
            batch.x = torch.cat([batch_reactant_x, batch_product_x], dim=-1)
        else:
            batch.x = self.pool(batch)

        if torch.isnan(batch.x).any():
            raise ValueError("Batch contains NaN values.")

        preds = self.head(batch)

        return {"preds": preds, 
                "reactant_features": reactant_features, 
                "product_features": product_features,
                "reactant_batch_indices": reactant_batch_indices,
                "product_batch_indices": product_batch_indices,}
