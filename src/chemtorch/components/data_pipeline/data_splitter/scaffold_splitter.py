import math
from typing import List
import warnings
import re
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict

try:
    # Python ≥ 3.12
    from typing import override  # type: ignore
except ImportError:
    # Python < 3.12
    from typing_extensions import override  # type: ignore

from chemtorch.components.data_pipeline.data_splitter import DataSplitterBase
from chemtorch.utils import DataSplit


class ScaffoldSplitter(DataSplitterBase):
    def __init__(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        include_chirality: bool = False,
        split_on: str | None = None,
        mol_idx: int | None = None,
        save_path: str | None = None,
    ):
        """
        Initializes the ScaffoldSplitter.

        Splits data by grouping molecules based on their Murcko scaffold, ensuring that
        all molecules with the same scaffold are in the same split (train, val, or test).
        This is a standard method to test a model's ability to generalize to new
        chemical scaffolds.

        Args:
            train_ratio (float): The desired ratio of data for the training set.
            val_ratio (float): The desired ratio of data for the validation set.
            test_ratio (float): The desired ratio of data for the test set.
            include_chirality (bool): If `True`, includes chirality in the scaffold generation.
            split_on (str | None): Specifies whether to use the 'reactant' or 'product' for
                scaffold generation when processing reaction SMILES. Required for reaction SMILES,
                ignored for single molecules. Defaults to None.
            mol_idx (int | None): Zero-based index specifying which molecule to use if multiple 
                are present (e.g., 'A.B>>C' or 'A.B'). Required when multiple molecules are present
                in the selected part of the reaction. Defaults to None.
            save_path (str | None, optional): If provided, saves split indices as pickle file.
        """
        super().__init__(save_path=save_path)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.split_on = split_on.lower() if split_on is not None else None
        self.mol_idx = mol_idx
        self.include_chirality = include_chirality

        ratio_sum = self.train_ratio + self.val_ratio + self.test_ratio
        if not math.isclose(ratio_sum, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"Ratios (train, val, test) must sum to 1.0, got {ratio_sum}")
        if split_on is not None and split_on.lower() not in ["reactant", "product"]:
            raise ValueError("`split_on` must be either 'reactant' or 'product' when provided.")
        if mol_idx is not None and (not isinstance(mol_idx, int) or mol_idx < 0):
            raise ValueError("`mol_idx` must be a non-negative integer when provided.")

    @override
    def _split(self, df: pd.DataFrame) -> DataSplit[List[int]]:
        """
        Splits the DataFrame based on molecular scaffolds.

        Args:
            df (pd.DataFrame): The input DataFrame to be split. Must contain a 'smiles' column
                              with either single molecule SMILES or reaction SMILES.

        Returns:
            DataSplit[List[int]]: A named tuple containing the train, val, and test indices.
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        if "smiles" not in df.columns:
            raise ValueError(
                f"SMILES column 'smiles' not found in DataFrame columns: {df.columns.tolist()}"
            )

        df_with_scaffold = df.copy()
        df_with_scaffold["_scaffold"] = df_with_scaffold["smiles"].apply(
            self._get_scaffold_smiles
        )

        has_scaffold_mask = df_with_scaffold["_scaffold"] != ""
        df_scaffolds = df_with_scaffold[has_scaffold_mask]
        no_scaffold_indices = df_with_scaffold.index[~has_scaffold_mask].tolist()

        if len(no_scaffold_indices) > 0:
            warnings.warn(
                f"{len(no_scaffold_indices)} molecules could not be scaffolded. "
                "They will be added to the training set."
            )

        scaffold_to_indices = defaultdict(list)
        for index, scaffold in df_scaffolds["_scaffold"].items():
            scaffold_to_indices[scaffold].append(index)

        scaffold_groups = list(scaffold_to_indices.values())
        # print(f"Found {len(scaffold_groups)} unique scaffolds.")
        scaffold_groups.sort(key=len, reverse=True)

        split_indices = {"train": no_scaffold_indices, "val": [], "test": []}
        n_total = len(df)
        target_sizes = {
            "train": self.train_ratio * n_total,
            "val": self.val_ratio * n_total,
            "test": self.test_ratio * n_total,
        }

        for group in scaffold_groups:
            # calculate how under-filled each split is (as a fraction of its target)
            needs = {}
            for name in split_indices:
                if target_sizes[name] > 0:
                    needs[name] = (
                        target_sizes[name] - len(split_indices[name])
                    ) / target_sizes[name]
                else:
                    needs[
                        name
                    ] = -np.inf  # don't assign to a split with a target size of 0

            # assign group to the split with the highest need
            best_split = max(needs, key=needs.get)
            split_indices[best_split].extend(group)

        train_df = df.loc[split_indices["train"]].sample(frac=1)
        val_df = df.loc[split_indices["val"]].sample(frac=1)
        test_df = df.loc[split_indices["test"]].sample(frac=1)

        # print("--- Scaffold Split Ratios ---")
        # print(
        #     f"Guidance: train={self.train_ratio:.3f}, val={self.val_ratio:.3f}, test={self.test_ratio:.3f}"
        # )
        # if n_total > 0:
        #     print(
        #         f"Actual:   train={len(train_df) / n_total:.3f}, val={len(val_df) / n_total:.3f}, test={len(test_df) / n_total:.3f}"
        #     )
        # print("-" * 29)

        indices = DataSplit(
            train=train_df.index.to_list(),
            val=val_df.index.to_list(),
            test=test_df.index.to_list(),
        )
        return indices

    def _get_scaffold_smiles(self, smiles: str) -> str:
        """
        Generates the Murcko scaffold SMILES for a specified molecule in a reaction or single molecule.

        Args:
            smiles (str): The reaction SMILES string (e.g., 'reactant>>product') or 
                         single molecule SMILES string.

        Returns:
            str: The SMILES string of the Murcko scaffold. Returns an empty string if
                 the molecule is invalid, has no scaffold (is acyclic), or cannot be found.
        """
        if pd.isna(smiles) or not isinstance(smiles, str) or smiles.strip() == "":
            raise ValueError(
                f"Invalid SMILES format: '{smiles}'."
            )

        # Check if it's a reaction SMILES (contains '>>')
        if ">>" in smiles:
            parts = smiles.split(">>")
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid reaction SMILES format: '{smiles}'. Expected 'reactant>>product'."
                )
            
            # For reaction SMILES, split_on is required
            if self.split_on is None:
                raise ValueError(
                    f"split_on parameter is required for reaction SMILES. "
                    "Please specify 'reactant' or 'product'."
                )
            
            target_smiles_group = parts[0] if self.split_on == "reactant" else parts[1]
        else:
            # Single molecule SMILES - split_on is ignored
            target_smiles_group = smiles

        mols_smiles = target_smiles_group.split(".")

        # If there are multiple molecules, mol_idx is required
        if len(mols_smiles) > 1 and self.mol_idx is None:
            raise ValueError(
                f"mol_idx parameter is required for multi-component SMILES: '{smiles}'. "
                f"Found {len(mols_smiles)} molecules, please specify which one to use (0-based index)."
            )
        
        # Use mol_idx if provided, otherwise default to 0 for single molecules
        mol_idx = self.mol_idx if self.mol_idx is not None else 0

        try:
            target_smiles = mols_smiles[mol_idx]
        except IndexError:
            raise IndexError(
                f"Molecule index {mol_idx} out of bounds for SMILES '{smiles}' "
                f"(has {len(mols_smiles)} molecules)."
            )

        mol = Chem.MolFromSmiles(target_smiles)
        if mol is None:
            raise ValueError(
                f"Could not parse molecule SMILES: '{target_smiles}'."
            )

        try:
            scaffold_smiles = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol, includeChirality=self.include_chirality
            )
            return scaffold_smiles
        except Exception:
            return ""

    def _remove_atom_mapping(self, smiles: str) -> str:
        """Removes atom map numbers (e.g., :1, :23) from a SMILES string."""
        return re.sub(r":\d+", "", smiles)

