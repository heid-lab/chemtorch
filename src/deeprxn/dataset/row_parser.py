from typing import Dict, Any
import pandas as pd

from deeprxn.data_pipeline.data_pipeline import DataPipelineComponent


class RowParser(DataPipelineComponent):
    """
    A utility class to parse rows of a DataFrame according to a specified configuration.

    Args:
        column_mapping (Dict[str, str]): A dictionary mapping representation arguments
            (e.g., "smiles", "label") to column names in the DataFrame.
    """

    def __init__(self, column_mapping: Dict[str, str]):
        """
        Initialize the RowParser.

        Args:
            column_mapping (Dict[str, str]): A mapping of argument names to column names.
                For example: {"smiles": "reaction_smiles", "label": "enthalpy"}
        """
        self.column_mapping = column_mapping

    def forward(self, row: pd.Series) -> Dict[str, Any]:
        """
        Parse a single row of the DataFrame.

        Args:
            row (pd.Series): A single row of the DataFrame.

        Returns:
            Dict[str, Any]: A dictionary of parsed values, where keys are argument names
            (e.g., "smiles", "label") and values are the corresponding values from the row.
        """
        parsed_values = {}
        for arg_name, column_name in self.column_mapping.items():
            if column_name not in row:
                raise KeyError(f"Column '{column_name}' not found in the DataFrame.")
            parsed_values[arg_name] = row[column_name]
        return parsed_values

