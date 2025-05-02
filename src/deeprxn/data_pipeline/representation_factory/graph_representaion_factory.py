from typing import Callable, Any
from torch_geometric.data import Data
from deeprxn.data_pipeline.data_pipeline import DataPipelineComponent


class GraphRepresentationFactory(DataPipelineComponent):
    """
    A factory class for creating graph representations.

    This class takes a partially instantiated representation (e.g., CGR) and
    uses it to create a PyTorch Geometric Data object for each sample.
    """

    def __init__(self, preconf_repr: Callable[..., Any]):
        """
        Initialize the GraphRepresentationFactory.

        Args:
            preconf_repr (Callable[..., Any]): A callable that takes keyword arguments
                and returns an instance of the representation.
        """
        self.preconf_repr = preconf_repr

    def forward(self, **kwargs) -> Data:
        """
        Create a new representation instance and convert it to a PyTorch Geometric Data object.

        Args:
            **kwargs: Keyword arguments to be passed to the representation.

        Returns:
            Data: A PyTorch Geometric Data object created from the representation.
        """
        representation = self.preconf_repr(**kwargs)
        return representation.to_pyg_data()