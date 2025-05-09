import pandas as pd
from torch import nn

from deepreaction.data_pipeline.data_split import DataSplit


class ColumnFilterAndRename(nn.Module):
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

    def forward(self, dataframes: DataSplit) -> DataSplit:
        """
        Process the DataSplit object by filtering and renaming columns in each DataFrame.

        Args:
            dataframes (DataSplit): A named tuple containing the train, val, and test DataFrames.
            Each DataFrame will be processed to filter and rename columns based on the column mapping.

        Returns:
            DataSplit: A named tuple containing the processed train, val, and test DataFrames.

        Raises:
            TypeError: If the input is not a DataSplit object or if any of the DataFrames are not pandas DataFrames.
            KeyError: If any of the columns specified in the column mapping are not found in any of the DataFrame.
        """
        if not isinstance(dataframes, DataSplit):
            raise TypeError("Input must be a DataSplit object")

        return DataSplit(
            *map(self._parse_dataframe, dataframes)
        )


    def _parse_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Parse the DataFrame to filter and rename columns based on the column mapping.
        
        Args:
            data (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The processed DataFrame with filtered and renamed columns.
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