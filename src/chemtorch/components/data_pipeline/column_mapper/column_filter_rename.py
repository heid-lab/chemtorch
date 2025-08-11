from typing import Union

import pandas as pd
try:
    # Python ≥ 3.12
    from typing import override  # type: ignore
except ImportError:
    # Python < 3.12
    from typing_extensions import override  # type: ignore

from chemtorch.components.data_pipeline.column_mapper.column_mapper import ColumnMapper
from chemtorch.utils import DataSplit


class ColumnFilterAndRename(ColumnMapper):
    """
    A pipeline component that filters and renames columns in a DataFrame
    based on provided column mappings.
    
    Usage:
        mapper = ColumnFilterAndRename(smiles="smiles_column", label="target_column")
        # This will rename "smiles_column" to "smiles" and "target_column" to "label"
    """

    def __init__(self, **column_mappings):
        """
        Initialize the ColumnFilterAndRename.

        Args:
            **column_mappings: Keyword arguments where the key is the new column name
                              and the value is the existing column name in the DataFrame.
                              Example: smiles="smiles_column", label="target_column"
        """
        super(ColumnFilterAndRename, self).__init__()
        # Filter out None values and empty strings
        self.column_mapping = {k: v for k, v in column_mappings.items() 
                             if v is not None and v != ""}
        
        if not self.column_mapping:
            raise ValueError(
                "At least one column mapping must be provided. "
                "Example: ColumnFilterAndRename(smiles='smiles_column', label='target_column')"
            )

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
