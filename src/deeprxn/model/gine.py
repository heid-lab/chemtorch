import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch_geometric.data import Batch
from torch_geometric.nn.aggr import SumAggregation

from deeprxn.model.model_base import Model


class GINE(Model):
    """Custom model using configurable components."""

    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        hidden_channels: int,
        mpnn_depth: int,
        shared_weights: bool,
        encoder_cfg: DictConfig,
        mpnn_cfg: DictConfig,
        pool_cfg: DictConfig,
        head_cfg: DictConfig,
    ):
        """Initialize Custom model."""
        super().__init__()
        self.mpnn_depth = mpnn_depth

        self.encoder = hydra.utils.instantiate(encoder_cfg)

        self.layers = nn.ModuleList()
        if shared_weights:
            layer = hydra.utils.instantiate(mpnn_cfg)
            for _ in range(self.mpnn_depth):
                self.layers.append(layer)
        else:
            for _ in range(self.mpnn_depth):
                self.layers.append(hydra.utils.instantiate(mpnn_cfg))

        # self.aggregation = SumAggregation()

        # self.edge_to_node = nn.Linear(
        #     num_node_features + hidden_channels, hidden_channels
        # )

        self.pool = hydra.utils.instantiate(pool_cfg)
        self.head = hydra.utils.instantiate(head_cfg)

    def forward(self, batch: Batch) -> Batch:
        """Forward pass through Custom model."""

        batch = self.encoder(batch)

        for layer in self.layers:
            batch = layer(batch)

        # s = self.aggregation(batch.h, batch.edge_index[1])

        # batch.q = torch.cat([batch.x, s, batch.pos_enc], dim=1)
        # batch.x = F.relu(self.edge_to_node(batch.q))

        batch.x = self.pool(batch)
        preds = self.head(batch)

        return preds
