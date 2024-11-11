from abc import abstractmethod

from torch_geometric.data import Batch
from torch_geometric.nn import MessagePassing


class MPNNLayer(MessagePassing):
    """Base class for all graph neural network layers.

    All layers should inherit from this class and implement the forward method.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """Initialize the base layer.

        Args:
            in_channels: Number of input features
            out_channels: Number of output features
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    @abstractmethod
    def forward(self, batch: Batch) -> Batch:
        """Forward pass of the layer.

        Args:
            batch: PyG batch containing graph data

        Returns:
            Updated batch with new features
        """
        pass
