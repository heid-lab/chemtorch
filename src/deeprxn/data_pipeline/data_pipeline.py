import pandas as pd

from abc import ABC, abstractmethod
from typing import Any, List, NamedTuple


class DataSplit(NamedTuple):
    """
    A named tuple to hold the data splits for training, validation, and testing.
    """
    train: Any
    val: Any
    test: Any


# Actually, this class is not needed except for documentation purposes, 
# since all components could simply subclass nn.Modules.
class DataPipelineComponent(ABC):
    """
    An abstract base class for data pipeline components and pipelines.

    This interface is used for both individual components (e.g., data readers,
    preprocessors, splitters) and entire pipelines (e.g., SourceProcessingPipeline,
    DatasetBuildingPipeline). Pipelines themselves can be treated as components
    and used to construct higher-level pipelines.

    Methods:
        forward: Executes the functionality of the component or pipeline.
    """

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        The forward method is called to execute the component's or pipeline's functionality.

        Returns:
            Any: The output of the component or pipeline.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Calls the forward method when the component or pipeline is called.

        Returns:
            Any: The output of the component or pipeline.
        """
        return self.forward(*args, **kwargs)


class DataPipeline(DataPipelineComponent):
    """
    A general data pipeline that consists of multiple components.

    This class allows for the sequential execution of components, where the output
    of one component is passed as input to the next.
    """
    def __init__(self, components: List[DataPipelineComponent]):
        """
        Initializes the DataPipeline with a list of components.

        Args:
            components (List[DataPipelineComponent]): A list of data pipeline components.
        """
        self.components = components

    def forward(self, data) -> Any:
        """
        Executes the data pipeline by calling the forward method of each components in order.

        Args:
            data: The input data to be processed by the pipeline.
        Returns:
            Any: The final output of the pipeline after processing through all components.
        """
        for component in self.components:
            data = component.forward(data)
        return data
