from abc import abstractmethod
from typing import Callable, Dict, Literal

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

from deepreaction.representation.rxn_graph_base import AtomOriginType


class Pool(nn.Module):
    """Base class for all pooling implementations."""

    VALID_AGGR: set[str] = {"add", "mean", "max"}
    _AGGR_FNS: Dict[str, Callable] = {
        "add": global_add_pool,
        "mean": global_mean_pool,
        "max": global_max_pool,
    }

    def __init__(self, aggr: Literal["add", "mean", "max"] = "add"):
        super().__init__()
        if aggr not in self.VALID_AGGR:
            raise ValueError(
                f"Invalid aggr: {aggr}. Must be one of {self.VALID_AGGR}"
            )
        self.aggr = aggr
        self.pool_fn = self._AGGR_FNS[aggr]

    @abstractmethod
    def forward(self, batch: Batch) -> torch.Tensor:
        """Pool the node features in the batch.

        Args:
            batch: PyG batch with node features and batch assignments

        Returns:
            torch.Tensor: Pooled features
        """
        pass


class GlobalPool(Pool):
    """Global pooling across all nodes."""

    def __init__(
        self,
        aggr: Literal["add", "mean", "max"] = "add",
    ):
        super().__init__(aggr)

    def forward(self, batch: Batch) -> torch.Tensor:
        """Pool all nodes globally.

        Args:
            batch: PyG batch with x and batch attributes

        Returns:
            torch.Tensor: Globally pooled features
        """
        return self.pool_fn(batch.x, batch.batch)


class TypePool(Pool):
    """Type-specific pooling based on atom origin type."""

    TYPE_MAPPING = {
        "reactants": AtomOriginType.REACTANT,
        "products": AtomOriginType.PRODUCT,
        "dummy": AtomOriginType.DUMMY,
        "reactant_product": AtomOriginType.REACTANT_PRODUCT,
    }

    def __init__(
        self,
        pool_type: Literal[
            "reactants", "products", "dummy", "reactant_product"
        ],
        aggr: Literal["add", "mean", "max"] = "add",
    ):
        super().__init__(aggr)
        if pool_type not in self.TYPE_MAPPING:
            raise ValueError(
                f"Invalid pool_type: {pool_type}. Must be one of {list(self.TYPE_MAPPING.keys())}"
            )
        self.pool_type = pool_type
        self.atom_type = self.TYPE_MAPPING[pool_type]

    def forward(self, batch: Batch) -> torch.Tensor:
        """Pool nodes of specific type.

        Args:
            batch: PyG batch with x, batch, and atom_origin_type attributes

        Returns:
            torch.Tensor: Type-specific pooled features
        """
        mask = batch.atom_origin_type == self.atom_type.value
        return self.pool_fn(batch.x[mask], batch.batch[mask])
