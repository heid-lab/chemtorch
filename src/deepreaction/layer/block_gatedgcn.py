import hydra
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from omegaconf import DictConfig
from torch_geometric.data import Batch

from deepreaction.act.act import Activation


class BlockGatedGCNLayer(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        in_channels: int,
        out_channels: int,
        residual: bool,
        dropout: float,
        activation: str,
        ffn: bool,
        mpnn_cfg: DictConfig,
    ):
        super(BlockGatedGCNLayer, self).__init__()
        self.dropout = dropout
        self.ffn = ffn

        self.mpnn = hydra.utils.instantiate(mpnn_cfg)

        # TODO: make component
        if self.ffn:
            # if self.batch_norm:
            self.norm1_ffn = pyg_nn.norm.BatchNorm(hidden_channels)
            # if self.layer_norm:
            #     self.norm1_local = pyg_nn.norm.LayerNorm(hidden_channels)
            self.ff_linear1 = nn.Linear(hidden_channels, hidden_channels * 2)
            self.ff_linear2 = nn.Linear(hidden_channels * 2, hidden_channels)
            self.act_fn_ff = Activation(activation_type=activation)
            # if self.batch_norm:
            self.norm2_ffn = pyg_nn.norm.BatchNorm(hidden_channels)
            # if self.layer_norm:
            #     self.norm2_local = pyg_nn.norm.LayerNorm(hidden_channels)
            self.ff_dropout1 = nn.Dropout(dropout)
            self.ff_dropout2 = nn.Dropout(dropout)

    def forward(self, batch: Batch) -> Batch:
        batch = self.mpnn(batch)

        if self.ffn:
            pre_ffn = batch.x
            # if self.batch_norm:
            batch.x = self.norm1_ffn(batch.x)
            # if self.layer_norm:
            #     batch.x = self.norm1_local(batch.x)
            batch.x = self.ff_dropout1(
                self.act_fn_ff(self.ff_linear1(batch.x))
            )
            batch.x = self.ff_dropout2(self.ff_linear2(batch.x))

            batch.x = pre_ffn + batch.x

            # if self.batch_norm:
            batch.x = self.norm2_ffn(batch.x)
            # if self.layer_norm:
            #     batch.x = self.norm2_local(batch.x)

        return batch
