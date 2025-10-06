"""
Simplified dataset property calculation system.

This module provides a clean way to compute properties needed for model configuration
at runtime, with proper handling of partition-dependent vs partition-independent properties.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union, TYPE_CHECKING, Literal, cast
import logging

import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree

from chemtorch.components.representation.token.abstract_token_representation import AbstractTokenRepresentation
from chemtorch.core.dataset_base import DatasetBase
from chemtorch.utils.types import DatasetKey, LightningTask, PropertySource


logger = logging.getLogger(__name__)


def compute_property_with_dataset_handling(
    property_instance: 'DatasetProperty', 
    dataset: Union[DatasetBase, Dict[str, DatasetBase], List[DatasetBase]]
) -> Any:
    """
    Compute a property while handling edge cases where dataset might be a dict or list.
    
    For dict datasets (multiple named test datasets), computes the property for the first dataset.
    For list datasets (multiple test datasets), computes the property for the first dataset.
    For single datasets, computes the property directly.
    
    Args:
        property_instance: The property to compute
        dataset: The dataset(s) to compute the property for
        
    Returns:
        The computed property value
    """
    if isinstance(dataset, dict):
        # For dict of datasets, use the first one (usually the base "test" dataset)
        first_dataset = next(iter(dataset.values()))
        return property_instance.compute(first_dataset)
    elif isinstance(dataset, list):
        # For list of datasets, use the first one (usually the base dataset without transforms)
        return property_instance.compute(dataset[0])
    else:
        # Single dataset case
        return property_instance.compute(dataset)


def resolve_sources(source: PropertySource, tasks: List[LightningTask]) -> List[DatasetKey]:
    """Resolve a PropertySource into a list of DatasetKeys."""
    existing_sources: List[DatasetKey] = []
    if "fit" in tasks:
        existing_sources.append("train")
    if "validate" in tasks:
        existing_sources.append("val")
    if "test" in tasks:
        existing_sources.append("test")
    if "predict" in tasks:
        existing_sources.append("predict")

    if source == "all":
        return existing_sources
    elif source == "any":
        return existing_sources[:1] if existing_sources else []
    elif source in ["train", "val", "test", "predict"]:
        return [source]
    else:
        raise ValueError(f"Unknown PropertySource: {source}")


class DatasetProperty(ABC):
    """Base class for dataset properties that can be computed at runtime."""

    def __init__(self, name: str, source: PropertySource, log: bool=False, add_to_cfg: bool=False) -> None:
        self.name = name
        self.source = source
        self.log = log
        self.add_to_cfg = add_to_cfg

    @abstractmethod
    def compute(self, dataset: DatasetBase) -> Any:
        """Compute the property value from the dataset."""
        pass

class PrecomputeTime(DatasetProperty):
    def compute(self, dataset: DatasetBase) -> float:
        if not dataset.precompute_all:
            raise ValueError("Dataset must be precomputed to compute precompute_time")
        return float(dataset.precompute_time if dataset.precompute_time is not None else 0.0)

# Concrete implementations
class NumNodeFeatures(DatasetProperty):
    def compute(self, dataset: DatasetBase[Data, Any]) -> int:
        if len(dataset) == 0:
            return 0
        first_item = dataset[0]
        data = first_item[0] if isinstance(first_item, tuple) else first_item
        if isinstance(data, Data) and data.x is not None:
            return data.x.size(1)
        return 0


class NumEdgeFeatures(DatasetProperty):
    def compute(self, dataset: DatasetBase) -> int:
        if len(dataset) == 0:
            return 0
        first_item = dataset[0] 
        data = first_item[0] if isinstance(first_item, tuple) else first_item
        if isinstance(data, Data) and data.edge_attr is not None:
            return data.edge_attr.size(1)
        return 0


class FingerprintLength(DatasetProperty):
    def compute(self, dataset: DatasetBase) -> int:
        if len(dataset) == 0:
            return 0
        first_item = dataset[0]
        data = first_item[0] if isinstance(first_item, tuple) else first_item
        if isinstance(data, torch.Tensor):
            return data.shape[0]
        return 0
    
class VocabSize(DatasetProperty):
    def compute(self, dataset: DatasetBase[Any, AbstractTokenRepresentation]) -> int:
        # Try dataset.representation.vocab_size
        if dataset.representation and hasattr(dataset.representation, 'vocab_size'):
            return int(dataset.representation.vocab_size)  # type: ignore
        else:
            raise ValueError("Dataset representation does not have 'vocab_size' attribute")


class LabelMean(DatasetProperty):
    def compute(self, dataset: DatasetBase) -> float:
        if not dataset.has_labels:
            raise ValueError("Dataset must have labels to compute mean")
        return float(dataset.dataframe['label'].mean())


class LabelStd(DatasetProperty):
    def compute(self, dataset: DatasetBase) -> float:
        if not dataset.has_labels:
            raise ValueError("Dataset must have labels to compute std")
        return float(dataset.dataframe['label'].std())


class DegreeStatistics(DatasetProperty):
    def compute(self, dataset: DatasetBase) -> Dict[str, Union[int, List[int]]]:
        if not dataset.precompute_all:
            raise ValueError("Dataset must be precomputed to compute degree statistics")
        
        if len(dataset) == 0:
            return {"max_degree": 0, "degree_histogram": []}
        
        max_degree = -1
        degree_histogram = None
        
        for item in dataset:
            data = item[0] if isinstance(item, tuple) else item
            if not isinstance(data, Data):
                raise TypeError(f"Expected PyTorch Geometric Data, got {type(data)}")
            
            if data.edge_index is None:
                continue  # Skip graphs with no edges
                
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            max_degree = max(max_degree, int(d.max()))
            
            if degree_histogram is None:
                degree_histogram = torch.zeros(max_degree + 1, dtype=torch.long)
            elif max_degree >= degree_histogram.numel():
                new_size = max_degree + 1
                resized_histogram = torch.zeros(new_size, dtype=torch.long)
                resized_histogram[:degree_histogram.numel()] = degree_histogram
                degree_histogram = resized_histogram
            
            degree_histogram += torch.bincount(d, minlength=degree_histogram.numel())
        
        return {
            "max_degree": max_degree,
            "degree_histogram": degree_histogram.tolist() if degree_histogram is not None else [],
        }
