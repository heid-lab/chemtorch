import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, SparseTensor


class GCNConvWithEdges(MessagePassing):
   """GCN convolution with edge features.

   This implementation is based on the GCN layer from PyTorch Geometric
   but extended to incorporate edge features in the message passing.
   """

   def __init__(
       self,
       in_channels,
       out_channels,
       edge_dim=None,
       bias=True,
       add_self_loops=False,
       normalize=False,
       **kwargs,
   ):
       """Initialize the GCN convolution with edge features.

       Parameters
       ----------
       in_channels : int
           The input feature dimension.
       out_channels : int
           The output feature dimension.
       edge_dim : int, optional
           The edge feature dimension, by default None.
       bias : bool, optional
           Whether to use bias, by default True.
       add_self_loops : bool, optional
           Whether to add self loops, by default False.
       normalize : bool, optional
           Whether to normalize edge weights, by default False.

       """
       super(GCNConvWithEdges, self).__init__(aggr="add", **kwargs)

       self.in_channels = in_channels
       self.out_channels = out_channels
       self.edge_dim = edge_dim
       self.add_self_loops = add_self_loops
       self.normalize = normalize

       self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)

       if bias:
           self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
       else:
           self.register_parameter("bias", None)

       self.reset_parameters()

   def reset_parameters(self):
       self.lin.reset_parameters()
       if self.bias is not None:
           torch.nn.init.zeros_(self.bias)

   def forward(self, x, edge_index, edge_attr=None, edge_weight=None):
       if self.normalize:
           if isinstance(edge_index, Tensor):
               edge_index, edge_weight = gcn_norm(
                   edge_index,
                   edge_weight,
                   x.size(0),
                   False,
                   self.add_self_loops,
                   flow=self.flow,
               )
           elif isinstance(edge_index, SparseTensor):
               edge_index = gcn_norm(
                   edge_index,
                   edge_weight,
                   x.size(0),
                   False,
                   self.add_self_loops,
                   flow=self.flow,
               )

       x = self.lin(x)

       out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

       if self.bias is not None:
           out += self.bias

       return out

   def message(self, x_j, edge_attr):
       # Combine node and edge features
       return (x_j + edge_attr).relu()

   def __repr__(self):
       return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels})"


class GCNLayer(nn.Module):
   """GCN layer that can optionally use edge attributes."""

   def __init__(
       self,
       in_channels,
       out_channels,
       use_edge_attr=False,
   ):
       """Initialize the GCN layer.

       Parameters
       ----------
       in_channels : int
           The input feature dimension.
       out_channels : int
           The output feature dimension.
       use_edge_attr : bool, optional
           Whether to use edge attributes, by default False.

       """
       super().__init__()
       self.use_edge_attr = use_edge_attr

       if use_edge_attr:
           self.model = GCNConvWithEdges(
               in_channels, out_channels, edge_dim=in_channels
           )
       else:
           self.model = pyg_nn.GCNConv(in_channels, out_channels)

   def forward(self, batch):
       if self.use_edge_attr:
           batch.x = self.model(
               batch.x, batch.edge_index, edge_attr=batch.edge_attr
           )
       else:
           batch.x = self.model(batch.x, batch.edge_index)
       return batch
