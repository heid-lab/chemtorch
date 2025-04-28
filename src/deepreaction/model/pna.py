import hydra
import torch.nn as nn


class PNA(nn.Module):
   """PNA model."""

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
       """Initialize the PNA model.

       Parameters
       ----------
       num_node_features : int
           The number of input node features.
       num_edge_features : int
           The number of input edge features.
       hidden_channels : int
           The hidden layer dimension.
       depth : int
           The number of message passing layers.
       shared_weights : bool
           Whether to use the same layer weights across depths.
       encoder_cfg : DictConfig
           Configuration for node/edge feature encoders.
       layer_cfg : DictConfig
           Configuration for the message passing layers.
       pool_cfg : DictConfig
           Configuration for the graph pooling layer.
       head_cfg : DictConfig
           Configuration for the prediction head.
       dataset_precomputed : any, optional
           Precomputed dataset statistics for normalization, by default None.

       """
       super().__init__()
       self.depth = depth

       self.encoders = nn.ModuleList()
       for _, config in encoder_cfg.items():
           self.encoders.append(hydra.utils.instantiate(config))

       self.layers = nn.ModuleList()
       if shared_weights:
           layer = hydra.utils.instantiate(
               layer_cfg, dataset_precomputed=dataset_precomputed
           )
           for _ in range(self.depth):
               self.layers.append(layer)
       else:
           for _ in range(self.depth):
               self.layers.append(
                   hydra.utils.instantiate(
                       layer_cfg, dataset_precomputed=dataset_precomputed
                   )
               )

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
