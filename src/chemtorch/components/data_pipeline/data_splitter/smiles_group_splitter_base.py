from abc import abstractmethod
from typing import List, Dict
import pandas as pd
from abc import ABC
from collections import defaultdict

from chemtorch.components.data_pipeline.data_splitter.group_splitter_base import GroupSplitterBase

try:
    # Python ≥ 3.12
    from typing import override  # type: ignore
except ImportError:
    # Python < 3.12
    from typing_extensions import override  # type: ignore


class SMILESGroupSplitterBase(GroupSplitterBase, ABC):

    @override
    def _group_dataframe(self, df: pd.DataFrame) -> tuple[Dict[str, List[int]], List[int]]:
        """
        Group the DataFrame by SMILES-based groups.
        
        Args:
            df (pd.DataFrame): The input DataFrame to be grouped. Must contain a 'smiles' column
                              with either single molecule SMILES or reaction SMILES.
            
        Returns:
            tuple[Dict[str, List[int]], List[int]]: A tuple containing:
                - A dictionary mapping group identifiers to lists of DataFrame indices that belong to that group
                - A list of indices where group assignment failed (will be added to training set)
        """
        if "smiles" not in df.columns:
            raise ValueError(
                f"SMILES column 'smiles' not found in DataFrame columns: {df.columns.tolist()}"
            )

        df_with_group = df.copy()
        df_with_group["_group"] = df_with_group["smiles"].apply(self._make_group_id_from_smiles)

        # Use None to indicate ungrouped SMILES instead of empty strings
        has_group = df_with_group["_group"].notna()
        grouped_df = df_with_group[has_group]
        ungrouped_indices = df_with_group.index[~has_group].tolist()

        group_to_indices = defaultdict(list)
        for index, group in grouped_df["_group"].items():
            group_to_indices[group].append(index)

        return dict(group_to_indices), ungrouped_indices

    @abstractmethod
    def _make_group_id_from_smiles(self, smiles: str) -> str | None:
        """
        Generate a group identifier for the given SMILES string.
        
        Args:
            smiles (str): The SMILES string to generate a group identifier for.
            
        Returns:
            str | None: The group identifier string, or None if the SMILES cannot be grouped.
        """
        pass