import hydra
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

from deepreaction.act.act import Activation


class BlockGatedGCNLayer(nn.Module):
   """Block gated graph convolutional network layer."""
   
   def __init__(
       self,
       hidden_channels,
       in_channels,
       out_channels,
       residual,
       dropout,
       activation,
       ffn,
       mpnn_cfg,
   ):
       """Initialize the block gated GCN layer.
       
       Parameters
       ----------
       hidden_channels : int
           The hidden dimension size.
       in_channels : int
           The input feature dimension.
       out_channels : int
           The output feature dimension.
       residual : bool
           Whether to use residual connections.
       dropout : float
           Dropout probability.
       activation : str
           Activation function type.
       ffn : bool
           Whether to apply feed-forward network after message passing.
       mpnn_cfg : DictConfig
           Configuration for the message passing neural network.
           
       """
       super().__init__()
       self.dropout = dropout
       self.ffn = ffn

       self.mpnn = hydra.utils.instantiate(mpnn_cfg)

       if self.ffn:
           self.norm1_ffn = pyg_nn.norm.BatchNorm(hidden_channels)

           self.ff_linear1 = nn.Linear(hidden_channels, hidden_channels * 2)
           self.ff_linear2 = nn.Linear(hidden_channels * 2, hidden_channels)
           self.act_fn_ff = Activation(activation_type=activation)
           self.norm2_ffn = pyg_nn.norm.BatchNorm(hidden_channels)

           self.ff_dropout1 = nn.Dropout(dropout)
           self.ff_dropout2 = nn.Dropout(dropout)

   def forward(self, batch):
       batch = self.mpnn(batch)

       if self.ffn:
           pre_ffn = batch.x
           batch.x = self.norm1_ffn(batch.x)

           batch.x = self.ff_dropout1(
               self.act_fn_ff(self.ff_linear1(batch.x))
           )
           batch.x = self.ff_dropout2(self.ff_linear2(batch.x))

           batch.x = pre_ffn + batch.x

           batch.x = self.norm2_ffn(batch.x)

       return batch
