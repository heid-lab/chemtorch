import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch_geometric.nn.aggr import SumAggregation


class DMPNN(nn.Module):
   """Directed Message Passing Neural Network."""

   def __init__(
       self,
       num_node_features,
       num_edge_features,
       hidden_channels,
       depth,
       shared_weights,
       encoder_cfg,
       layer_cfg,
       pool_cfg,
       head_cfg,
   ):
       """Initialize the DMPNN model.
       
       Parameters
       ----------
       num_node_features : int
           The input node feature dimension.
       num_edge_features : int
           The input edge feature dimension.
       hidden_channels : int
           The hidden layer dimension.
       depth : int
           The number of message passing layers.
       shared_weights : bool
           Whether to use shared weights across layers.
       encoder_cfg : DictConfig
           Configuration for node/edge encoders.
       layer_cfg : DictConfig
           Configuration for message passing layers.
       pool_cfg : DictConfig
           Configuration for graph pooling.
       head_cfg : DictConfig
           Configuration for output head.
           
       """
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

       self.aggregation = SumAggregation()

       self.edge_to_node = nn.Linear(
           num_node_features + hidden_channels, hidden_channels
       )

       self.pool = hydra.utils.instantiate(pool_cfg)
       self.head = hydra.utils.instantiate(head_cfg)

   def forward(self, batch):
       for encoder in self.encoders:
           batch = encoder(batch)

       for layer in self.layers:
           batch = layer(batch)

       s = self.aggregation(batch.h, batch.edge_index[1])

       batch.q = torch.cat([batch.x, s], dim=1)
       batch.x = F.relu(self.edge_to_node(batch.q))

       batch.x = self.pool(batch)
       preds = self.head(batch)

       return preds
