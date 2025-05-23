from typing import Any, Callable, Dict, Literal, Optional, Union
import torch.nn as nn
from torch_geometric.data import Batch

from deepreaction.layer.gnn_block import GNNBlockLayer
from deepreaction.layer.mpnn_layer.dmpnn_layer import DMPNNLayer
from deepreaction.layer.utils import normalize


class DMPNNBlockLayer(GNNBlockLayer):
    def __init__(
        self,
        mpnn: DMPNNLayer,
        use_residual: bool = False,
        use_ffn: bool = False,
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        hidden_channels: int = None,
    ):
        super(DMPNNBlockLayer, self).__init__(
            mpnn=mpnn,
            use_mpnn_residual=use_residual,
            use_ffn=use_ffn,
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
        self.mpnn_residual.register(batch.h_0)
        # Message passing
        batch = self.mpnn(batch)

        # Optional normalization
        batch.h = normalize(batch.h, batch, self.mpnn_norm)
        # Activation
        batch.h = self.activation(batch.h)

        # Optional residual connection
        batch.h = self.mpnn_residual.apply(batch.h)

        # Optional Feed-Forward Network (FFN)
        if self.use_ffn:
            self.ffn_residual.register(batch.h)
            batch.h = normalize(batch.h, batch, self.ffn_norm_in)
            batch.h = self.ffn(batch.h)
            batch.h = self.dropout(batch.h)
            batch.h = self.ffn_residual.apply(batch.h)
            batch.h = normalize(batch.h, batch, self.ffn_norm_out)

        return batch


    