from typing import Dict, List, Union, Protocol, TYPE_CHECKING
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree

if TYPE_CHECKING:
    from chemtorch.components.dataset.abstact_dataset import AbstractDataset

class GraphRepresentationProtocol(Protocol):
    """Protocol for representations that produce PyTorch Geometric Data objects."""
    
    def construct(self, **kwargs) -> Data: ...
    
    def __call__(self, **kwargs) -> Data: ...

def compute_degree_statistics(dataset: AbstractDataset[Data, GraphRepresentationProtocol]) -> Dict[str, Union[int, List[int]]]:
    """
    Compute degree statistics for a graph dataset.
    
    Args:
        dataset: A dataset that produces Data objects
        
    Returns:
        Dictionary with max_degree and degree_histogram
        
    Raises:
        ValueError: If dataset is not precomputed
        TypeError: If dataset doesn't contain Data objects
    """
    if not dataset.precompute_all:
        raise ValueError("Dataset must be precomputed to compute degree statistics.")
    
    if len(dataset) == 0:
        return {"max_degree": 0, "degree_histogram": []}
    
    first_item = dataset[0]
    if isinstance(first_item, tuple):
        first_data = first_item[0]
    else:
        first_data = first_item
        
    if not isinstance(first_data, Data):
        raise TypeError(f"Expected Data objects, got {type(first_data)}")

    max_degree = -1
    degree_histogram: torch.Tensor | None = None

    for item in dataset:
        if isinstance(item, tuple):
            data = item[0]
        else:
            data = item
            
        if data.edge_index is None:
            continue
            
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

def get_num_node_features(dataset: AbstractDataset[Data, GraphRepresentationProtocol]) -> int:
    """Get number of node features from a graph dataset."""
    if len(dataset) == 0:
        raise ValueError("Cannot determine node features from empty dataset")
    
    first_item = dataset[0]
    if isinstance(first_item, tuple):
        data = first_item[0]
    else:
        data = first_item
    
    if not isinstance(data, Data):
        raise TypeError(f"Expected Data objects, got {type(data)}")
    
    return data.num_node_features

def get_num_edge_features(dataset: AbstractDataset[Data, GraphRepresentationProtocol]) -> int:
    """Get number of edge features from a graph dataset."""
    if len(dataset) == 0:
        raise ValueError("Cannot determine edge features from empty dataset")
    
    first_item = dataset[0]
    if isinstance(first_item, tuple):
        data = first_item[0]
    else:
        data = first_item
    
    if not isinstance(data, Data):
        raise TypeError(f"Expected Data objects, got {type(data)}")
    
    return data.num_edge_features
