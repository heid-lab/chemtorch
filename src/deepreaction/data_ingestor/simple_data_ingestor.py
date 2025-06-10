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
        Executes the data ingestion pipeline with validation.

        Returns:
            DataSplit: A named tuple containing the train, val, and test DataFrames.

        Raises:
            ValueError: If there is a configuration mismatch, such as:
                        - A `data_splitter` is provided for a pre-split dataset.
                        - A single DataFrame is loaded but no `data_splitter` is provided.
            TypeError: If the components produce unexpected data types at any stage.
        """
        # load data
        raw_data = self.data_source.load()

        # map columns
        processed_data = self.column_mapper(raw_data)

        if isinstance(processed_data, pd.DataFrame):
            # case: data is a single DataFrame, so a splitter is required
            if self.data_splitter is None:
                raise ValueError(
                    "Data is a single DataFrame, but no 'data_splitter' was provided "
                    "to split it into train, validation, and test sets."
                )
            final_data_split = self.data_splitter(processed_data)

        elif isinstance(processed_data, DataSplit):
            # case: data is already split. a splitter is redundant
            if self.data_splitter is not None:
                raise ValueError(
                    "The data is already split (presplit dataset), but a 'data_splitter' "
                    "was also provided. Please provide one or the other, not both."
                )
            final_data_split = processed_data

        else:
            # case: the data is of an unexpected type
            raise TypeError(
                f"The data after column mapping has an unexpected type: {type(processed_data).__name__}. "
                f"Expected a pandas DataFrame or a DataSplit object."
            )

        if not isinstance(final_data_split, DataSplit):
            raise TypeError(
                f"The final output of the ingestion pipeline is not a DataSplit object, "
                f"but a {type(final_data_split).__name__}. There might be an issue with the data_splitter."
            )

        return final_data_split
