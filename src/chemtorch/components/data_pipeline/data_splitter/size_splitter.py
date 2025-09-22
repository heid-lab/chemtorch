import math
import pandas as pd
from typing import List

from chemtorch.components.data_pipeline.data_splitter.ratio_splitter import RatioSplitter

try:
    # Python ≥ 3.12
    from typing import override  # type: ignore
except ImportError:
    # Python < 3.12
    from typing_extensions import override  # type: ignore

from chemtorch.utils import DataSplit
from chemtorch.utils.atom_mapping import make_mol


class SizeSplitter(RatioSplitter):
    """
    Splits data into training, validation, and test sets based on molecular size (number
    of heavy atoms). For single molecules, sorts by the number of heavy atoms in each molecule.
    For reactions, sorts by the sum of heavy atoms in reactants and products.
    """
    def __init__(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        sort_order: str = "ascending",  # ascending is train small, test large
        save_path: str | None = None,
    ):
        """
        Initializes the SizeSplitter.

        Args:
            train_ratio (float): The ratio of data for the training set.
            val_ratio (float): The ratio of data for the validation set.
            test_ratio (float): The ratio of data for the test set.
            sort_order (str): 'ascending' or 'descending'.
            save_path (str | None, optional): If provided, saves split indices as pickle file.
        """
        super().__init__(train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, save_path=save_path)
        self.sort_order = sort_order.lower()
        if self.sort_order not in ["ascending", "descending"]:
            raise ValueError("sort_order must be 'ascending' or 'descending'.")

    @override
    def _split(self, df: pd.DataFrame) -> DataSplit[List[int]]:
        """
        Splits the DataFrame based on molecular size (number of heavy atoms).

        For single molecules: sorts by the number of heavy atoms in each molecule.
        For reactions: sorts by the sum of heavy atoms in reactants and products.
        
        The DataFrame is first augmented with a molecule size column, then sorted by this size.
        It's subsequently split into train, validation, and test sets.
        Finally, each set is shuffled randomly.

        Args:
            df (pd.DataFrame): The input DataFrame to be split. Must contain the 'smiles' column
                              with either single molecule SMILES or reaction SMILES.

        Returns:
            DataSplit[List[int]]: A named tuple containing the train, val, and test indices.
        """
        if "smiles" not in df.columns:
            raise ValueError(
                f"SMILES column 'smiles' not found in DataFrame columns: {df.columns.tolist()}"
            )

        df_with_size = df.copy()
        
        # Calculate molecular size with error handling
        mol_sizes = []
        for idx, smiles in df_with_size["smiles"].items():
            try:
                mol_size = self._get_n_heavy_atoms(smiles)
                mol_sizes.append(mol_size)
            except Exception as e:
                raise ValueError(f"Error processing SMILES at row index {idx}: {str(e)}") from e
        
        df_with_size["_mol_size"] = mol_sizes


        is_ascending = self.sort_order == "ascending"
        sorted_indices = (
            df_with_size["_mol_size"].sort_values(ascending=is_ascending).index
        )
        df_sorted = df_with_size.loc[sorted_indices]

        n_total = len(df_sorted)
        train_size = int(self.train_ratio * n_total)
        val_size = int(self.val_ratio * n_total)

        train_df = df_sorted.iloc[:train_size].sample(frac=1)
        val_df = df_sorted.iloc[train_size : train_size + val_size].sample(
            frac=1
        )
        test_df = df_sorted.iloc[train_size + val_size :].sample(frac=1)

        indices = DataSplit(
            train=train_df.index.to_list(),
            val=val_df.index.to_list(),
            test=test_df.index.to_list(),
        )

        return indices

    def _get_n_heavy_atoms(self, smiles: str | None) -> int:
        """
        Calculates the number of heavy atoms in a molecule or reaction from its SMILES string.
        
        For single molecules: returns the number of heavy atoms in the molecule.
        For reactions (containing '>>'): returns the sum of heavy atoms in reactants and products.
        """

        if pd.isna(smiles) or not isinstance(smiles, str) or smiles.strip() == "":
            raise ValueError("Invalid SMILES string.")

        # Check if it's a reaction SMILES (contains '>>')
        if ">>" in smiles:
            parts = smiles.split(">>")
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid reaction SMILES format: '{smiles}'. "
                    "Expected 'reactant>>product'."
                )
            smiles_reac, smiles_prod = parts

            mol_reac, _ = make_mol(smiles_reac)
            mol_prod, _ = make_mol(smiles_prod)

            return mol_reac.GetNumHeavyAtoms() + mol_prod.GetNumHeavyAtoms()
        else:
            # Single molecule SMILES
            mol, _ = make_mol(smiles)
            return mol.GetNumHeavyAtoms()