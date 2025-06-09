# deepreaction/data_ingestor/simple_data_ingestor.py (new file)
from typing import Optional

import pandas as pd

from deepreaction.data_ingestor.column_mapper.column_mapper import ColumnMapper
from deepreaction.data_ingestor.data_source.data_source import DataSource
from deepreaction.data_ingestor.data_splitter.data_splitter import DataSplitter
from deepreaction.utils import DataSplit


class SimpleDataIngestor:
    """
    A simple data ingestor that orchestrates data loading, column mapping,
    and data splitting.

    The ingestion process is as follows:
    1. Load data using the `data_source`. This can result in a single
       DataFrame or an already split `DataSplit` object.
    2. Apply column transformations (filtering, renaming) using the `column_mapper`.
       This mapper can operate on both single DataFrames and `DataSplit` objects.
    3. If the data after mapping is a single DataFrame, split it using the
       `data_splitter`. If it's already a `DataSplit`, this step is skipped.
    """

    def __init__(
        self,
        data_source: DataSource,
        column_mapper: ColumnMapper,
        data_splitter: Optional[DataSplitter] = None,
    ):
        """
        Initializes the SimpleDataIngestor.

        Args:
            data_source (DataSource): The component responsible for loading the initial data.
            column_mapper (ColumnMapper): The component for transforming columns.
                                              It should handle both pd.DataFrame and DataSplit inputs.
            data_splitter (Optional[DataSplitter]): The component for splitting a single DataFrame
                                                    into train, validation, and test sets.
                                                    This is not used if data_source already provides split data.
        """
        self.data_source = data_source
        self.column_mapper = column_mapper
        self.data_splitter = data_splitter

    def __call__(self) -> DataSplit:
        """
        Executes the data ingestion pipeline.

        Returns:
            DataSplit: A named tuple containing the train, val, and test DataFrames.

        Raises:
            ValueError: If data_splitter is required but not provided.
            TypeError: If the components produce unexpected data types.
        """

        # load data
        raw_data = self.data_source.load()

        # map columns
        processed_data = self.column_mapper(raw_data)

        # split data if necessary
        if self.data_splitter is not None:
            final_data_split = self.data_splitter(processed_data)
        else:
            final_data_split = processed_data

        return final_data_split
