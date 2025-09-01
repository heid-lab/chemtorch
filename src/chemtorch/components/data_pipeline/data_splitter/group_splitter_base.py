from typing import List, Dict, Any
import warnings
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

from chemtorch.components.data_pipeline.data_splitter.ratio_splitter import RatioSplitter

try:
    # Python ≥ 3.12
    from typing import override  # type: ignore
except ImportError:
    # Python < 3.12
    from typing_extensions import override  # type: ignore

from chemtorch.utils import DataSplit

class GroupSplitterBase(RatioSplitter, ABC):
    """
    Base class for data splitters that group data points by a common property.
    
    This abstract class provides the common logic for splitting data by first grouping
    it based on some property (e.g., scaffold, reaction core, etc.) and then distributing
    the groups across train/val/test splits to maintain the desired ratios.
    
    Subclasses must implement the `_group_dataframe` method to define how to group the data.
    """
    
    @abstractmethod
    def _group_dataframe(self, df: pd.DataFrame) -> tuple[Dict[Any, List[int]], List[int]]:
        """
        Group the DataFrame by some property and return both grouped and ungrouped indices.
        
        This method must be implemented by subclasses to define how to group the data.
        For example, scaffold splitters would group by molecular scaffold, while
        reaction core splitters might group by reaction core.
        
        Args:
            df (pd.DataFrame): The input DataFrame to be grouped.
            
        Returns:
            tuple[Dict[Any, List[int]], List[int]]: A tuple containing:
                - A dictionary mapping group keys to lists of DataFrame indices that belong to that group
                - A list of indices that could not be grouped (will be added to training set by default)
                                 
        Raises:
            Any exceptions related to the specific grouping logic should be raised here.
        """
        pass
    
    def _determine_split_for_ungrouped_indices(self) -> str:
        """
        Determine which split ungrouped indices should be assigned to.
        
        By default, ungrouped indices are assigned to the training set to maintain
        consistency with historical behavior. Subclasses can override this method
        to implement different strategies.
            
        Returns:
            str: The split name ('train', 'val', or 'test') to assign ungrouped indices to.
        """
        return "train"
    
    @override
    def _split(self, df: pd.DataFrame) -> DataSplit[List[int]]:
        """
        Split the DataFrame by grouping data points and distributing groups across splits.
        
        Args:
            df (pd.DataFrame): The input DataFrame to be split.
            
        Returns:
            DataSplit[List[int]]: A named tuple containing the train, val, and test indices.
        """
        
        # Group the data using the subclass-specific grouping logic
        group_to_indices, ungrouped_indices = self._group_dataframe(df)
        if len(ungrouped_indices) > 0:
            warnings.warn(
                f"{len(ungrouped_indices)} molecules could not be assigned to any group. "
                f"They will be added to the {self._determine_split_for_ungrouped_indices()} set."
            )
        
        # Convert to list of groups and sort by size (largest first)
        groups = list(group_to_indices.values())
        groups.sort(key=len, reverse=True)
        
        # Initialize splits
        split_indices = {"train": [], "val": [], "test": []}
        n_total = len(df)
        split_sizes = {
            "train": self.train_ratio * n_total,
            "val": self.val_ratio * n_total,
            "test": self.test_ratio * n_total,
        }
        
        # Handle ungrouped indices (typically add to training set)
        if ungrouped_indices:
            target_split = self._determine_split_for_ungrouped_indices()
            split_indices[target_split].extend(ungrouped_indices)
        
        # Distribute groups across splits to match target ratios
        for group in groups:
            # Calculate how under-filled each split is (as a fraction of its target)
            needs = {}
            for split_name in split_indices:
                if split_sizes[split_name] > 0:
                    needs[split_name] = (
                        split_sizes[split_name] - len(split_indices[split_name])
                    ) / split_sizes[split_name]
                else:
                    needs[split_name] = -np.inf  # don't assign to a split with a target size of 0
            
            # Assign group to the split with the highest need
            best_split = max(needs.keys(), key=lambda x: needs[x])
            split_indices[best_split].extend(group)
        
        train_df = df.loc[split_indices["train"]]
        val_df = df.loc[split_indices["val"]]
        test_df = df.loc[split_indices["test"]]

        indices = DataSplit(
            train=train_df.index.to_list(),
            val=val_df.index.to_list(),
            test=test_df.index.to_list(),
        )
        return indices