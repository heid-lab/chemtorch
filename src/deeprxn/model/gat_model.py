from dataclasses import dataclass

import hydra
import torch
import torch.nn as nn
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from torch_geometric.data import Batch

from deeprxn.head.head import FFNHead
from deeprxn.model.model_base import Model
from deeprxn.mpnn_layer.gat_layer import GATLayer
from deeprxn.mpnn_layer.mpnn_layer_base import Layer
from deeprxn.pool.pool import GlobalPool


class GATModel(Model):
    """Graph Attention Network model with configurable components."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        layer_config: DictConfig,
        pool_config: DictConfig,
        head_config: DictConfig,
    ):
        """Initialize GAT model."""
        super().__init__(
            in_channels, hidden_channels, out_channels, num_layers
        )

        self.layers = nn.ModuleList(
            [
                hydra.utils.instantiate(
                    layer_config,
                    in_channels=in_channels if i == 0 else hidden_channels,
                    out_channels=hidden_channels,
                )
                for i in range(num_layers)
            ]
        )

        self.pool = hydra.utils.instantiate(pool_config)
        self.head = hydra.utils.instantiate(
            head_config,
            in_channels=hidden_channels,
            out_channels=out_channels,
        )

    def forward(self, batch: Batch) -> Batch:
        """Forward pass through GAT model."""
        for layer in self.layers:
            batch = layer(batch)

        batch = self.pool(batch)
        batch.pred = self.head(batch)

        return batch
