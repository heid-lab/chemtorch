from typing import List, Dict
import warnings
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

from chemtorch.components.data_pipeline.data_splitter.smiles_group_splitter_base import SMILESGroupSplitterBase


class ScaffoldSplitter(SMILESGroupSplitterBase):
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
        super().__init__(train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, save_path=save_path)
        self.split_on = split_on.lower() if split_on is not None else None
        self.mol_idx = mol_idx
        self.include_chirality = include_chirality

        if split_on is not None and split_on.lower() not in ["reactant", "product"]:
            raise ValueError("`split_on` must be either 'reactant' or 'product' when provided.")
        if mol_idx is not None and (not isinstance(mol_idx, int) or mol_idx < 0):
            raise ValueError("`mol_idx` must be a non-negative integer when provided.")

    @override
    def _make_group_id_from_smiles(self, smiles: str) -> str | None:
        """
        Generates the Murcko scaffold SMILES for a specified molecule in a reaction or single molecule.

        Args:
            smiles (str): The reaction SMILES string (e.g., 'reactant>>product') or 
                         single molecule SMILES string.

        Returns:
            str | None: The SMILES string of the Murcko scaffold, or None if
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
            # Return None for empty scaffolds (acyclic molecules) instead of empty string
            return scaffold_smiles if scaffold_smiles else None
        except Exception:
            return None
