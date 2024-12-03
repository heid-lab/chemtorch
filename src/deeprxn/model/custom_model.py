import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch_geometric.data import Batch
from torch_geometric.nn.aggr import SumAggregation

from deeprxn.model.model_base import Model


class CustomModel(Model):
    """Custom model using configurable components."""

    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        hidden_channels: int,
        mpnn_depth: int,
        att_depth: int,
        shared_weights: bool,
        encoder_cfg: DictConfig,
        mpnn_cfg: DictConfig,
        att_layer_cfg: DictConfig,
        pool_cfg: DictConfig,
        head_cfg: DictConfig,
    ):
        """Initialize Custom model."""
        super().__init__()
        self.mpnn_depth = mpnn_depth
        self.att_depth = att_depth

        self.encoder = hydra.utils.instantiate(encoder_cfg)

        self.mpnn_layers = nn.ModuleList()
        if shared_weights:
            mpnn_layer = hydra.utils.instantiate(mpnn_cfg)
            for _ in range(self.mpnn_depth):
                self.mpnn_layers.append(mpnn_layer)
        else:
            for _ in range(self.mpnn_depth):
                self.mpnn_layers.append(hydra.utils.instantiate(mpnn_cfg))

        self.aggregation = SumAggregation()

        self.edge_to_node = nn.Linear(
            num_node_features + hidden_channels, hidden_channels
        )

        self.att_layers = nn.ModuleList()
        if shared_weights:
            att_layer = hydra.utils.instantiate(att_layer_cfg)
            for _ in range(self.att_depth):
                self.att_layers.append(att_layer)
        else:
            for _ in range(self.att_depth):
                self.att_layers.append(hydra.utils.instantiate(att_layer_cfg))

        self.pool = hydra.utils.instantiate(pool_cfg)
        self.head = hydra.utils.instantiate(head_cfg)

    def forward(self, batch: Batch) -> Batch:
        """Forward pass through Custom model."""

        batch = self.encoder(batch)

        for mpnn_layer in self.mpnn_layers:
            batch = mpnn_layer(batch)

        s = self.aggregation(batch.h, batch.edge_index[1])

        batch.q = torch.cat([batch.x, s], dim=1)
        batch.x = F.relu(self.edge_to_node(batch.q))

        for att_layer in self.att_layers:
            batch = att_layer(batch)

        batch.x = self.pool(batch)
        preds = self.head(batch)

        return preds
