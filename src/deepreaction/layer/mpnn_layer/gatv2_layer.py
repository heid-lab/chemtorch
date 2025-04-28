import torch.nn as nn
from torch_geometric.nn import GATv2Conv


class GATv2Layer(nn.Module):
   """Graph Attention Network Layer wrapper."""

   def __init__(
       self,
       in_channels,
       out_channels,
       heads=1,
       dropout=0.0,
       concat=True,
       use_edge_attr=True,
   ):
       """Initialize GAT layer.

       Parameters
       ----------
       in_channels : int
           Number of input features.
       out_channels : int
           Number of output features.
       heads : int, optional
           Number of attention heads, by default 1.
       dropout : float, optional
           Dropout probability, by default 0.0.
       concat : bool, optional
           Whether to concatenate or average multi-head attention outputs, by default True.
       use_edge_attr : bool, optional
           Whether to use edge attributes, by default True.

       """
       super().__init__()

       self.use_edge_attr = use_edge_attr

       self.gat = GATv2Conv(
           in_channels=in_channels,
           out_channels=out_channels,
           heads=heads,
           dropout=dropout,
           concat=concat,
           edge_dim=in_channels if use_edge_attr else None,
       )

   def forward(self, batch):
       if self.use_edge_attr:
           batch.x = self.gat(batch.x, batch.edge_index, batch.edge_attr)
       else:
           batch.x = self.gat(batch.x, batch.edge_index)
       return batch
