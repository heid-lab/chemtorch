import hydra
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

from deepreaction.act.act import Activation


class BlockGCNLayer(nn.Module):
   """Block Graph Convolutional Network layer."""

   def __init__(
       self,
       hidden_channels,
       in_channels,
       out_channels,
       layer_norm,
       batch_norm,
       residual,
       dropout,
       activation,
       ffn,
       mpnn_cfg,
   ):
       """Initialize the Block GCN layer.

       Parameters
       ----------
       hidden_channels : int
           The hidden feature dimension.
       in_channels : int
           The input feature dimension.
       out_channels : int
           The output feature dimension.
       layer_norm : bool
           Whether to use layer normalization.
       batch_norm : bool
           Whether to use batch normalization.
       residual : bool
           Whether to use residual connections.
       dropout : float
           Dropout probability.
       activation : str
           Activation function type.
       ffn : bool
           Whether to use feed-forward network after message passing.
       mpnn_cfg : DictConfig
           Configuration for the message passing neural network.

       """
       super().__init__()

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

   def forward(self, batch):
       if self.residual:
           pre_layer = batch.x

       batch = self.mpnn(batch)

       if self.layer_norm:
           batch.x = self.norm(batch.x, batch.batch)
       if self.batch_norm:
           batch.x = self.norm(batch.x)

       if self.activation:
           batch.x = self.activation(batch.x)

       if self.dropout > 0:
           batch.x = F.dropout(
               batch.x, p=self.dropout, training=self.training
           )

       if self.residual:
           batch.x = batch.x + pre_layer

       if self.ffn:
           pre_ffn = batch.x
           if self.batch_norm:
               batch.x = self.norm1_ffn(batch.x)
           if self.layer_norm:
               batch.x = self.norm1_local(batch.x)
           batch.x = self.ff_dropout1(
               self.act_fn_ff(self.ff_linear1(batch.x))
           )
           batch.x = self.ff_dropout2(self.ff_linear2(batch.x))

           batch.x = pre_ffn + batch.x

           if self.batch_norm:
               batch.x = self.norm2_ffn(batch.x)
           if self.layer_norm:
               batch.x = self.norm2_local(batch.x)

       return batch
