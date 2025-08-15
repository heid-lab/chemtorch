import math
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

from chemtorch.components.data_pipeline.data_splitter import DataSplitter
from chemtorch.utils import DataSplit


class ScaffoldSplitter(DataSplitter):
    def __init__(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        split_on: str = "reactant",
        mol_idx: str | int = "first",
        include_chirality: bool = False,
        save_split_dir: str | None = None,
        save_indices: bool = True,
        save_csv: bool = False,
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
            split_on (str): Specifies whether to use the 'reactant' or 'product' for
                scaffold generation. Defaults to 'reactant'.
            mol_idx (str | int): Specifies which molecule to use if multiple are present
                (e.g., 'A.B>>C'). Can be 'first', 'last', or a zero-based integer index.
                Defaults to 'first'.
            include_chirality (bool): If `True`, includes chirality in the scaffold generation.
            save_split_dir (str | None, optional): If provided, enables saving of split files.
            save_indices (bool): If True and `save_split_dir` is set, saves 'indices.pkl'.
            save_csv (bool): If True and `save_split_dir` is set, saves split DataFrames as CSVs.
        """
        super().__init__(
            save_split_dir=save_split_dir, save_indices=save_indices, save_csv=save_csv
        )
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.split_on = split_on.lower()
        self.mol_idx = mol_idx
        self.include_chirality = include_chirality

        ratio_sum = self.train_ratio + self.val_ratio + self.test_ratio
        if not math.isclose(ratio_sum, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"Ratios (train, val, test) must sum to 1.0, got {ratio_sum}")
        if self.split_on not in ["reactant", "product"]:
            raise ValueError("`split_on` must be either 'reactant' or 'product'.")
        if not (isinstance(self.mol_idx, int) or self.mol_idx in ["first", "last"]):
            raise ValueError("`mol_idx` must be an integer, 'first', or 'last'.")

    def _get_scaffold_smiles(self, smiles: str) -> str:
        """
        Generates the Murcko scaffold SMILES for a specified molecule in a reaction.

        Args:
            smiles (str): The reaction SMILES string (e.g., 'reactant>>product').

        Returns:
            str: The SMILES string of the Murcko scaffold. Returns an empty string if
                 the molecule is invalid, has no scaffold (is acyclic), or cannot be found.
        """
        if pd.isna(smiles) or not isinstance(smiles, str) or ">>" not in smiles:
            warnings.warn(
                f"Invalid reaction SMILES format: '{smiles}'. Assigning no scaffold."
            )
            return ""

        parts = smiles.split(">>")
        target_smiles_group = parts[0] if self.split_on == "reactant" else parts[1]
        mols_smiles = target_smiles_group.split(".")

        try:
            if self.mol_idx == "first":
                selected_smiles = mols_smiles[0]
            elif self.mol_idx == "last":
                selected_smiles = mols_smiles[-1]
            else:
                selected_smiles = mols_smiles[self.mol_idx]
        except IndexError:
            warnings.warn(
                f"Molecule index {self.mol_idx} out of bounds for SMILES '{smiles}'. Assigning no scaffold."
            )
            return ""

        non_atom_mapped_selected_smiles = self._remove_atom_map_number_manual(
            selected_smiles
        )
        mol = Chem.MolFromSmiles(non_atom_mapped_selected_smiles)
        if mol is None:
            warnings.warn(
                f"Could not parse molecule SMILES: '{selected_smiles}'. Assigning no scaffold."
            )
            return ""

        try:
            scaffold_smiles = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol, includeChirality=self.include_chirality
            )
            return scaffold_smiles
        except Exception as e:
            warnings.warn(
                f"Failed to generate scaffold for '{selected_smiles}' due to: {e}. Assigning no scaffold."
            )
            return ""

    def _remove_atom_map_number_manual(self, smiles: str) -> str:
        """Removes atom map numbers (e.g., :1, :23) from a SMILES string."""
        return re.sub(r":\d+", "", smiles)

    @override
    def __call__(self, df: pd.DataFrame) -> DataSplit:
        """
        Splits the DataFrame based on molecular scaffolds.

        Args:
            df (pd.DataFrame): The input DataFrame to be split. Must contain a 'smiles' column.

        Returns:
            DataSplit: A named tuple containing the train, val, and test DataFrames.
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
        df_has_scaffold = df_with_scaffold[has_scaffold_mask]
        no_scaffold_indices = df_with_scaffold.index[~has_scaffold_mask].tolist()

        if len(no_scaffold_indices) > 0:
            warnings.warn(
                f"{len(no_scaffold_indices)} molecules could not be scaffolded. "
                "They will be added to the training set."
            )

        scaffold_to_indices = defaultdict(list)
        for index, scaffold in df_has_scaffold["_scaffold"].items():
            scaffold_to_indices[scaffold].append(index)

        scaffold_groups = list(scaffold_to_indices.values())
        print(f"Found {len(scaffold_groups)} unique scaffolds.")
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

        train_indices, val_indices, test_indices = (
            split_indices["train"],
            split_indices["val"],
            split_indices["test"],
        )

        train_df = df.loc[train_indices].sample(frac=1)
        val_df = df.loc[val_indices].sample(frac=1)
        test_df = df.loc[test_indices].sample(frac=1)

        n_total = len(df)
        print("--- Scaffold Split Ratios ---")
        print(
            f"Guidance: train={self.train_ratio:.3f}, val={self.val_ratio:.3f}, test={self.test_ratio:.3f}"
        )
        if n_total > 0:
            print(
                f"Actual:   train={len(train_df) / n_total:.3f}, val={len(val_df) / n_total:.3f}, test={len(test_df) / n_total:.3f}"
            )
        print("-" * 29)

        data_split = DataSplit(
            train=train_df.reset_index(drop=True),
            val=val_df.reset_index(drop=True),
            test=test_df.reset_index(drop=True),
        )

        self._save_split(
            data_split=data_split,
            train_indices=train_df.index,
            val_indices=val_df.index,
            test_indices=test_df.index,
        )

        return data_split
