from typing import Any, Callable, Dict, Optional, Union

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn.resolver import (
    activation_resolver,
    aggregation_resolver,
)

from chemtorch.layer.layer_stack import LayerStack


class EdgeToNodeEmbedding(nn.Module):
    """
    EdgeToNodeEmbedding is a neural network layer that takes edge embeddings and node features,
    aggregates the edge embeddings based on the node indices, concatenates them with the node features,
    and passes them through a linear layer followed by an activation function to produce node embeddings.
    """

    def __init__(
        self,
        embedding_size: int,
        num_node_features: int,
        aggr: Union[str, Callable] = "add",
        aggr_kwargs: Optional[Dict[str, Any]] = None,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the EdgeToNodeEmbedding layer.

        Args:
            embedding_size (int): Size of the edge embeddings (also size of the new node embeddings).
            aggr (Union[str, Callable], optional): Aggregation method. Defaults to "add".
            aggr_kwargs (Optional[Dict[str, Any]], optional): Additional arguments for aggregation. Defaults to None.
            act (Union[str, Callable, None], optional): Activation function. Defaults to "relu".
            act_kwargs (Optional[Dict[str, Any]], optional): Additional arguments for activation function. Defaults to None.
        """
        super(EdgeToNodeEmbedding, self).__init__()

        self.linear = nn.Linear(
            in_features=num_node_features + embedding_size,
            out_features=embedding_size,
        )
        self.activation = activation_resolver(act, **(act_kwargs or {}))
        self.aggregation = aggregation_resolver(aggr, **(aggr_kwargs or {}))

    def forward(self, batch: Batch) -> Batch:
        """
        Forward pass through the EdgeToNodeEmbedding layer.

        Args:
            batch (Batch): The input batch of graphs containing node features and edge embeddings.

        Returns:
            Batch: The output batch with updated node features.
        """
        h_aggr = self.aggregation(
            batch.h, batch.edge_index[1], dim_size=batch.num_nodes
        )
        batch.q = torch.cat([batch.x, h_aggr], dim=1)
        batch.x = self.linear(batch.q)
        batch.x = self.activation(batch.x)
        return batch


class DMPNNStack(nn.Module):
    """
    DMPNNStack is a neural network layer that implements a sequence of directed message passing steps,
    followed by an edge-to-node embedding layer, which generates node embeddings from the original node
    features and the edge embeddings obtained from the directed message passing steps.
    """

    def __init__(
        self,
        dmpnn_blocks: LayerStack[Batch],
        edge_to_node_embedding: EdgeToNodeEmbedding,
    ):
        """
        Initialize the DMPNNStack.

        Args:
            dmpnn_blocks (Stack[DMPNNBlock]): A stack of DMPNN blocks that perform directed message passing.
            edge_to_node_embedding (EdgeToNodeEmbedding): The layer that converts edge features to node features.
        """
        super(DMPNNStack, self).__init__()
        self.dmpnn_blocks = dmpnn_blocks
        self.edge_to_node_embedding = edge_to_node_embedding

    def forward(self, batch: Batch) -> torch.Tensor:
        """
        Forward pass through the DMPNN layer.

        Args:
            batch (Batch): The input batch of graphs.

        Returns:
            torch.Tensor: The output predictions.
        """
        batch = self.dmpnn_blocks(batch)
        batch = self.edge_to_node_embedding(batch)
        return batch
