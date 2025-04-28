import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import Linear as Linear_pyg


class GINELayer(nn.Module):
   """Graph Isomorphism Network with Edge features (GINE) layer."""

   def __init__(
       self,
       in_channels,
       out_channels,
   ):
       """Initialize the GINE layer.

       Parameters
       ----------
       in_channels : int
           The input feature dimension.
       out_channels : int
           The output feature dimension.

       """
       super().__init__()
       gin_nn = nn.Sequential(
           Linear_pyg(in_channels, out_channels),
           nn.ReLU(),
           Linear_pyg(out_channels, out_channels),
       )
       self.model = pyg_nn.GINEConv(gin_nn)

   def forward(self, batch):
       batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr)
       return batch
