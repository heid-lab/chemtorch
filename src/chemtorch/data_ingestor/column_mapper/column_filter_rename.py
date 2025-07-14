from typing import Union

import pandas as pd
from typing_extensions import override

from chemtorch.data_ingestor.column_mapper.column_mapper import ColumnMapper
from chemtorch.utils import DataSplit


class ColumnFilterAndRename(ColumnMapper):
    """
    A pipeline component that filters and renames columns in a DataFrame
    based on a provided column mapping.
    """

    def __init__(self, column_mapping: dict):
        """
        Initialize the ColumnFilterAndRename.

        Args:
            column_mapping (dict): A dictionary mapping new column names to
                                   existing column names in the DataFrame.
        """
        super(ColumnFilterAndRename, self).__init__()
        self.column_mapping = column_mapping

    @override
    def __call__(
        self, data: Union[DataSplit, pd.DataFrame]
    ) -> Union[DataSplit, pd.DataFrame]:
        """
        Process the input data to filter and rename columns based on the column mapping.

        Args:
            data (Union[DataSplit, pd.DataFrame]): The input data, which can be a DataSplit object
                containing multiple DataFrames or a single pandas DataFrame.

        Returns:
            Union[DataSplit, pd.DataFrame]: The processed data with filtered and renamed columns.
                If the input is a DataSplit, it returns a DataSplit with each DataFrame processed.
                If the input is a pandas DataFrame, it returns a single processed DataFrame.

        Raises:
            TypeError: If the input is not a DataSplit object or a pandas DataFrame.
            KeyError: If any of the columns specified in the column mapping are not found in any of the DataFrame.
        """
        if isinstance(data, pd.DataFrame):
            return self._parse_dataframe(data)
        elif isinstance(data, DataSplit):
            return DataSplit(*map(self._parse_dataframe, data))
        else:
            raise TypeError(
                "Input must be a pandas DataFrame or a DataSplit object containing DataFrames"
            )

    def _parse_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Parse the DataFrame to filter and rename columns based on the column mapping.

        Args:
            data (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The processed DataFrame with filtered and renamed columns.

        Raises:
            TypeError: If the input is not a pandas DataFrame.
            KeyError: If any of the columns specified in the column mapping are not found in the DataFrame.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        missing_columns = [
            col for col in self.column_mapping.values() if col not in data.columns
        ]
        if missing_columns:
            raise KeyError(
                f"The following columns are missing from the DataFrame: {missing_columns}"
            )

        renamed_data = data[list(self.column_mapping.values())].rename(
            columns={v: k for k, v in self.column_mapping.items()}
        )
        return renamed_data
