import copy
from abc import ABC
from typing import Any

from torch_geometric.data import Data


# code inspired from https://pytorch-geometric.readthedocs.io/en/2.5.3/_modules/torch_geometric/transforms/base_transform.html#BaseTransform
class TransformBase(ABC):
    """Base class for graph transformations. Transformations take in Data objects and return Data objects."""

    def __call__(self, data: Data) -> Data:
        # shallow copy to prevent in-place modification
        return self.forward(copy.copy(data))

    def forward(self, data: Data) -> Data:
        pass
