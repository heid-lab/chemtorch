# started from code from https://github.com/rampasek/GraphGPS/tree/main, MIT License, Copyright (c) 2022 Ladislav Rampášek, Michael Galkin, Vijay Prakash Dwivedi, Dominique Beaini
import hydra
import torch.nn as nn


class GPS(nn.Module):
   """GPS model."""

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
       dataset_precomputed=None,
   ):
       """Initialize the GPS model.

       Parameters
       ----------
       num_node_features : int
           The number of input node features.
       num_edge_features : int
           The number of input edge features.
       hidden_channels : int
           The dimension of hidden representations.
       depth : int
           The number of message passing layers.
       shared_weights : bool
           Whether to share weights across layers.
       encoder_cfg : DictConfig
           Configuration for node/edge encoders.
       layer_cfg : DictConfig
           Configuration for message passing layers.
       pool_cfg : DictConfig
           Configuration for graph pooling.
       head_cfg : DictConfig
           Configuration for prediction head.
       dataset_precomputed : object, optional
           Precomputed dataset features, by default None.

       """
       super().__init__()
       self.depth = depth

       self.encoders = nn.ModuleList()
       for _, config in encoder_cfg.items():
           self.encoders.append(hydra.utils.instantiate(config))

       self.layers = nn.ModuleList()
       if shared_weights:
           if dataset_precomputed:
               layer = hydra.utils.instantiate(
                   layer_cfg, dataset_precomputed=dataset_precomputed
               )
           else:
               layer = hydra.utils.instantiate(layer_cfg)
           for _ in range(self.depth):
               self.layers.append(layer)
       else:
           for _ in range(self.depth):
               if dataset_precomputed:
                   layer = hydra.utils.instantiate(
                       layer_cfg, dataset_precomputed=dataset_precomputed
                   )
               else:
                   layer = hydra.utils.instantiate(layer_cfg)
               self.layers.append(layer)

       self.pool = hydra.utils.instantiate(pool_cfg)
       self.head = hydra.utils.instantiate(head_cfg)

   def forward(self, batch):
       for encoder in self.encoders:
           batch = encoder(batch)

       for layer in self.layers:
           batch = layer(batch)

       batch.x = self.pool(batch)
       preds = self.head(batch)

       return preds
