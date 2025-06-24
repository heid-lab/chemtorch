from typing import Callable, Dict, Literal

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import (
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

from chemtorch.representation.graph.graph_reprs_utils import AtomOriginType

AGGR_FNS: Dict[str, Callable] = {
    "add": global_add_pool,
    "mean": global_mean_pool,
    "max": global_max_pool,
}


class GlobalPool(nn.Module):
    """
    Global pooling across all nodes in a batch.
    """

    def __init__(self, aggr: Literal["add", "mean", "max"] = "add"):
        super().__init__()
        if aggr not in AGGR_FNS:
            raise ValueError(
                f"Invalid aggr: {aggr}. Must be one of {list(AGGR_FNS.keys())}"
            )
        self.pool_fn = AGGR_FNS[aggr]

    def forward(self, batch: Batch) -> torch.Tensor:
        return self.pool_fn(batch.x, batch.batch)


class AtomTypePool(nn.Module):
    """
    Type-specific pooling based on atom origin type.
    """

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
        super().__init__()
        if aggr not in AGGR_FNS:
            raise ValueError(
                f"Invalid aggr: {aggr}. Must be one of {list(AGGR_FNS.keys())}"
            )
        if pool_type not in self.TYPE_MAPPING:
            raise ValueError(
                f"Invalid pool_type: {pool_type}. Must be one of {list(self.TYPE_MAPPING.keys())}"
            )
        self.pool_fn = AGGR_FNS[aggr]
        self.atom_type = self.TYPE_MAPPING[pool_type]

    def forward(self, batch: Batch) -> torch.Tensor:
        mask = batch.atom_origin_type == self.atom_type.value
        return self.pool_fn(batch.x[mask], batch.batch[mask])
