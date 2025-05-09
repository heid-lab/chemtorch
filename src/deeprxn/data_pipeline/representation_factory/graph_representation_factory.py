from typing import Callable, Any, Dict
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

    def forward(self, sample: Dict[str, Any]) -> Data:
        """
        Create a new representation instance and convert it to a PyTorch Geometric Data object.

        Args:
            sample (Dict[str, Any]): A dictionary containing the data to instantiate the
                preconfigured representation.
                The keys should match the expected arguments of the representation's constructor.

        Returns:
            Data: A PyTorch Geometric Data object created from the representation.
        """
        representation = self.preconf_repr(**sample)
        graph = representation.to_pyg_data()
        return graph