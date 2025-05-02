from deeprxn.data_pipeline.data_pipeline import DataPipelineComponent


from abc import abstractmethod


class DataReader(DataPipelineComponent):
    """
    Abstract base class for data readers.
    This class defines the interface for reading data from various sources.
    """

    @abstractmethod
    def forward(self):
        raise NotImplementedError("Subclasses should implement this method.")