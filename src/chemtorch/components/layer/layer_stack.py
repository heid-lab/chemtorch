from collections.abc import Callable, Mapping
from copy import deepcopy
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
        self,
        layer: DictConfig | Mapping | Callable[[], nn.Module] | nn.Module,
        depth: int,
        share_weights: bool = False,
    ):
        """
        Initialize the Stack using a layer config, factory, or module.

        Args:
            layer: The layer to stack. This can be a Hydra/OmegaConf config,
                a mapping config, a zero-argument factory such as
                ``functools.partial(MyLayer, ...)``, or an instantiated module.
            depth (int): The number of times to repeat the layer.
            share_weights (bool): If True, share weights between the stacked layers.
        """
        super(LayerStack, self).__init__()
        self.layers = nn.ModuleList()

        def make_layer() -> nn.Module:
            if isinstance(layer, nn.Module):
                return layer if share_weights else deepcopy(layer)
            if isinstance(layer, (DictConfig, Mapping)):
                return safe_instantiate(layer)
            if callable(layer):
                new_layer = layer()
                if not isinstance(new_layer, nn.Module):
                    raise TypeError(
                        "Layer factory must return a torch.nn.Module, "
                        f"got {type(new_layer)}."
                    )
                return new_layer
            raise TypeError(
                "layer must be a config, a torch.nn.Module, or a zero-argument "
                f"factory returning a torch.nn.Module, got {type(layer)}."
            )

        if share_weights:
            single_layer = make_layer()
            for _ in range(depth):
                self.layers.append(single_layer)
        else:
            for _ in range(depth):
                self.layers.append(make_layer())

    def forward(self, x: T) -> T:
        for layer in self.layers:
            x = layer(x)
        return x
