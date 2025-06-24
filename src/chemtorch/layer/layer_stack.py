from typing import Generic, TypeVar

from omegaconf import DictConfig
from torch import nn

from chemtorch.utils.hydra import safe_instantiate

T = TypeVar("T")


class LayerStack(nn.Module, Generic[T]):
    """
    A utility class for stacking a layer multiple times.

    This class is useful for creating deep neural networks by stacking
    the same layer multiple times.

    Note, that the input and output types of the layer must be the same.
    """

    def __init__(
        self, layer: DictConfig, depth: int, share_weights: bool = False
    ):
        """
        Initialize the Stack using Hydra for instantiation.

        Args:
            layer (DictConfig): The configuration for the layer to be stacked.
            depth (int): The number of times to repeat the layer.
            share_weights (bool): If True, share weights between the stacked layers.
        """
        super(LayerStack, self).__init__()
        self.layers = nn.ModuleList()
        if share_weights:
            single_layer = safe_instantiate(layer)
            for _ in range(depth):
                self.layers.append(single_layer)
        else:
            for _ in range(depth):
                new_layer = safe_instantiate(layer)
                self.layers.append(new_layer)

    def forward(self, x: T) -> T:
        for layer in self.layers:
            x = layer(x)
        return x
