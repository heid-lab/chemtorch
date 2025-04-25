import hydra
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from omegaconf import DictConfig
from torch_geometric.data import Batch

from deepreaction.act.act import Activation, ActivationType
from deepreaction.layer.mpnn_layer.mpnn_layer_base import MPNNLayerBase


class BlockDMPNNLayer(MPNNLayerBase):
    def __init__(
        self,
        hidden_channels: int,
        in_channels: int,
        out_channels: int,
        layer_norm: bool,
        batch_norm: bool,
        residual: bool,
        dropout: float,
        activation: str,
        ffn: bool,
        mpnn_cfg: DictConfig,
    ):
        MPNNLayerBase.__init__(self, in_channels, out_channels)

        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.residual = residual
        self.dropout = dropout
        self.ffn = ffn

        self.mpnn = hydra.utils.instantiate(mpnn_cfg)

        if layer_norm and batch_norm:
            raise ValueError(
                "Only one of layer_norm and batch_norm can be True."
            )

        if self.layer_norm:
            self.norm = pyg_nn.norm.LayerNorm(hidden_channels)

        if self.batch_norm:
            self.norm = pyg_nn.norm.BatchNorm(hidden_channels)

        self.activation = Activation(activation_type=activation)

        # TODO: make component
        if self.ffn:
            if self.batch_norm:
                self.norm1_ffn = pyg_nn.norm.BatchNorm(hidden_channels)
            if self.layer_norm:
                self.norm1_local = pyg_nn.norm.LayerNorm(hidden_channels)
            self.ff_linear1 = nn.Linear(hidden_channels, hidden_channels * 2)
            self.ff_linear2 = nn.Linear(hidden_channels * 2, hidden_channels)
            self.act_fn_ff = Activation(activation_type=activation)
            if self.batch_norm:
                self.norm2_ffn = pyg_nn.norm.BatchNorm(hidden_channels)
            if self.layer_norm:
                self.norm2_local = pyg_nn.norm.LayerNorm(hidden_channels)
            self.ff_dropout1 = nn.Dropout(dropout)
            self.ff_dropout2 = nn.Dropout(dropout)

    def forward(self, batch: Batch) -> Batch:
        batch = self.mpnn(batch)

        if self.layer_norm:
            batch.h = self.norm(batch.h, batch.batch)
        if self.batch_norm:
            batch.h = self.norm(batch.h)

        if self.activation:
            batch.h = self.activation(batch.h)

        if self.dropout > 0:
            batch.h = F.dropout(
                batch.h, p=self.dropout, training=self.training
            )

        if self.residual:
            batch.h = batch.h + batch.h_0

        if self.ffn:
            pre_ffn = batch.h
            if self.batch_norm:
                batch.h = self.norm1_ffn(batch.h)
            if self.layer_norm:
                batch.h = self.norm1_local(batch.h)
            batch.h = self.ff_dropout1(
                self.act_fn_ff(self.ff_linear1(batch.h))
            )
            batch.h = self.ff_dropout2(self.ff_linear2(batch.h))

            batch.h = pre_ffn + batch.h

            if self.batch_norm:
                batch.h = self.norm2_ffn(batch.h)
            if self.layer_norm:
                batch.h = self.norm2_local(batch.h)

        return batch
