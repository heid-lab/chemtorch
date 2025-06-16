from typing import Any, Callable, Dict, Literal, Optional, Union

import torch.nn as nn
from torch_geometric.data import Batch

from deepreaction.layer.gnn_layer.gnn_block.gnn_block import GNNBlock
from deepreaction.layer.gnn_layer.graph_conv.dmpnn_conv import DMPNNConv
from deepreaction.layer.utils import normalize


class DMPNNBlock(GNNBlock):
    def __init__(
        self,
        graph_conv: DMPNNConv,
        residual: bool = False,
        ffn: bool = False,
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        hidden_channels: int = None,
    ):
        super(DMPNNBlock, self).__init__(
            graph_conv=graph_conv,
            residual=residual,
            ffn=ffn,
            dropout=dropout,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            hidden_channels=hidden_channels,
        )

    # override
    def forward(self, batch: Batch) -> Batch:
        # Register original input features for residual connection
        self.residual.register(batch.h_0)
        # Message passing
        batch = self.graph_conv(batch)

        # Optional normalization
        batch.h = normalize(batch.h, batch, self.norm)
        # Activation
        batch.h = self.activation(batch.h)

        # Optional dropout
        if self.dropout is not None:
            batch.h = self.dropout(batch.h)

        # Optional residual connection
        batch.h = self.residual.apply(batch.h)

        # Optional Feed-Forward Network (FFN)
        if self.use_ffn:
            self.ffn_residual.register(batch.h)
            batch.h = normalize(batch.h, batch, self.ffn_norm_in)
            batch.h = self.ffn_dropout1(
                self.ffn_act_fn(self.ffn_linear1(batch.h))
            )
            batch.h = self.ffn_dropout2(self.ffn_linear2(batch.h))
            batch.h = self.dropout(batch.h)
            batch.h = self.ffn_residual.apply(batch.h)
            batch.h = normalize(batch.h, batch, self.ffn_norm_out)

        return batch
