import hydra
import torch.nn as nn
from omegaconf import DictConfig
from torch_geometric.data import Batch


class GAT(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        hidden_channels: int,
        depth: int,
        shared_weights: bool,
        encoder_cfg: DictConfig,
        layer_cfg: DictConfig,
        pool_cfg: DictConfig,
        head_cfg: DictConfig,
    ):
        super().__init__()
        self.depth = depth

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
        for encoder in self.encoders:
            batch = encoder(batch)

        for layer in self.layers:
            batch = layer(batch)

        batch.x = self.pool(batch)
        preds = self.head(batch)

        return preds
