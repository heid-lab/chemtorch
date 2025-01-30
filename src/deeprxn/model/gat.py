import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from omegaconf import DictConfig
from torch_geometric.data import Batch
from torch_geometric.nn.aggr import SumAggregation

from deeprxn.model.model_base import Model


class GAT(Model):
    """Custom model using configurable components."""

    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        hidden_channels: int,
        depth: int,
        layer_norm: bool,
        batch_norm: bool,
        shared_weights: bool,
        encoder_cfg: DictConfig,
        layer_cfg: DictConfig,
        pool_cfg: DictConfig,
        head_cfg: DictConfig,
    ):
        """Initialize Custom model."""
        super().__init__()
        self.depth = depth
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        if layer_norm == batch_norm:
            raise ValueError(
                "Only one of layer_norm and batch_norm can be True."
            )

        if self.layer_norm:
            self.norm = pyg_nn.norm.LayerNorm(hidden_channels)

        if self.batch_norm:
            self.norm = nn.BatchNorm1d(hidden_channels)

        self.encoders = nn.ModuleList()
        for _, config in encoder_cfg.items():
            self.encoders.append(hydra.utils.instantiate(config))

        self.layers = nn.ModuleList()
        if shared_weights:
            layer = hydra.utils.instantiate(layer_cfg)
            for _ in range(self.depth):
                self.layers.append(layer)
        else:
            for _ in range(self.depth):
                self.layers.append(hydra.utils.instantiate(layer_cfg))

        self.pool = hydra.utils.instantiate(pool_cfg)
        self.head = hydra.utils.instantiate(head_cfg)

    def forward(self, batch: Batch) -> Batch:
        """Forward pass through Custom model."""

        for encoder in self.encoders:
            batch = encoder(batch)

        for layer in self.layers:
            pre_layer = batch.x
            batch = layer(batch)
            batch.x = batch.x + pre_layer
            if self.layer_norm:
                batch.x = self.norm(batch.x, batch.batch)
            if self.batch_norm:
                batch.x = self.norm(batch.x)

        batch.x = self.pool(batch)
        preds = self.head(batch)

        return preds
