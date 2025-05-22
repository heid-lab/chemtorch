from typing import Any, Callable, Dict, Optional, Union

from torch import nn
import torch
from torch_geometric.data import Batch
from torch_geometric.nn.resolver import activation_resolver, aggregation_resolver

class DMPNNLastLayer(nn.Module):
    """
    Last layer of the DMPNN model.
    
    This layer aggregates edge embeddings output from the DMPNN layers and combines them with the node features.
    It does so by concatenating the aggregated edge features with the node features,
    followed by a linear transformation and an activation function.
    """
    def __init__(
        self,
        num_node_features: int,
        hidden_channels: int,
        aggr: Union[str, Callable] = "add",
        aggr_kwargs: Optional[Dict[str, Any]] = None,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the last layer of the DMPNN model.
        
        Args:
            num_node_features (int): Number of node features.
            hidden_channels (int): Number of hidden channels.
            aggr (Union[str, Callable], optional): Aggregation method. Defaults to "add".
            aggr_kwargs (Optional[Dict[str, Any]], optional): Additional arguments for aggregation. Defaults to None.
            act (Union[str, Callable, None], optional): Activation function. Defaults to "relu".
            act_kwargs (Optional[Dict[str, Any]], optional): Additional arguments for activation function. Defaults to None.
        """
        super(DMPNNLastLayer, self).__init__()

        self.linear = nn.Linear(
            in_features=num_node_features + hidden_channels, 
            out_features=hidden_channels
        )
        self.activation = activation_resolver(act, **(act_kwargs or {}))
        self.aggregation = aggregation_resolver(aggr, **(aggr_kwargs or {}))

    def forward(self, batch: Batch) -> Batch:
        h_aggr = self.aggregation(batch.h, batch.edge_index[1])
        batch.q = torch.cat([batch.x, h_aggr], dim=1)
        batch.x = self.linear(batch.q)
        batch.x = self.activation(batch.x)
        return batch
