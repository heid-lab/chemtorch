from deeprxn.data_pipeline.data_pipeline import DataPipelineComponent, DataSplit


from abc import abstractmethod


class DataSplitter(DataPipelineComponent):
    """
    Abstract base class for data splitting strategies.
    """

    @abstractmethod
    def forward(self, raw) -> DataSplit:
        """
        Splits the raw data into training, validation, and test partitions.

        Args:
            raw: The raw data to be split.

        Returns:
            DataSplit: A named tuple containing the train, val, and test dataframes.
        """
        raise NotImplementedError("Subclasses should implement this method.")