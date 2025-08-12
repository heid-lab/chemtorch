import pandas as pd

try:
    # Python ≥ 3.12
    from typing import override  # type: ignore
except ImportError:
    # Python < 3.12
    from typing_extensions import override  # type: ignore

from chemtorch.components.data_pipeline.data_splitter import DataSplitter
from chemtorch.utils import DataSplit
from chemtorch.utils.atom_mapping import make_mol


class SizeSplitter(DataSplitter):
    def __init__(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        sort_order: str = "ascending",  # ascending is train small, test large
        save_split_dir: str | None = None,
        save_indices: bool = True,
        save_csv: bool = False,
    ):
        """
        Initializes the SizeSplitter.

        Args:
            train_ratio (float): The ratio of data for the training set.
            val_ratio (float): The ratio of data for the validation set.
            test_ratio (float): The ratio of data for the test set.
            sort_order (str): 'ascending' or 'descending'.
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
        self.sort_order = sort_order.lower()

        if not (
            1 - 1e-4 < self.train_ratio + self.val_ratio + self.test_ratio < 1 + 1e-4
        ):
            raise ValueError("Ratios (train, val, test) must sum to approximately 1.")
        if self.sort_order not in ["ascending", "descending"]:
            raise ValueError("sort_order must be 'ascending' or 'descending'.")

    def _get_n_heavy_atoms(self, smiles: str) -> int:
        """
        Calculates the number of heavy atoms in a molecule from its SMILES string.
        """

        if pd.isna(smiles) or not isinstance(smiles, str):
            raise ValueError("Invalid SMILES string.")

        smiles_reac, _, smiles_prod = smiles.split(">")

        mol_reac, _ = make_mol(smiles_reac)
        mol_prod, _ = make_mol(smiles_prod)

        return mol_reac.GetNumHeavyAtoms() + mol_prod.GetNumHeavyAtoms()

    @override
    def __call__(self, df: pd.DataFrame) -> DataSplit:
        """
        Splits the DataFrame based on molecular size (number of heavy atoms).

        The DataFrame is first augmented with a molecule size column, then sorted by this size.
        It's subsequently split into train, validation, and test sets.
        Finally, each set is shuffled randomly.

        Args:
            df (pd.DataFrame): The input DataFrame to be split. Must contain the 'smiles' column.

        Returns:
            DataSplit: A named tuple containing the train, val, and test DataFrames.
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        if "smiles" not in df.columns:
            raise ValueError(
                f"SMILES column 'smiles' not found in DataFrame columns: {df.columns.tolist()}"
            )

        df_with_size = df.copy()
        df_with_size["_mol_size"] = df_with_size["smiles"].apply(
            self._get_n_heavy_atoms
        )

        is_ascending = self.sort_order == "ascending"
        sorted_indices = (
            df_with_size["_mol_size"].sort_values(ascending=is_ascending).index
        )
        df_sorted = df_with_size.loc[sorted_indices]

        n_total = len(df_sorted)
        train_size = round(self.train_ratio * n_total)
        val_size = round(self.val_ratio * n_total)

        train_df_partition = df_sorted.iloc[:train_size]
        val_df_partition = df_sorted.iloc[train_size : train_size + val_size]
        test_df_partition = df_sorted.iloc[train_size + val_size :]

        data_split = DataSplit(
            train=train_df_partition.sample(frac=1)
            .reset_index(drop=True)
            .drop(columns=["_mol_size"]),
            val=val_df_partition.sample(frac=1)
            .reset_index(drop=True)
            .drop(columns=["_mol_size"]),
            test=test_df_partition.sample(frac=1)
            .reset_index(drop=True)
            .drop(columns=["_mol_size"]),
        )

        self._save_split(
            data_split=data_split,
            train_indices=train_df_partition.index,
            val_indices=val_df_partition.index,
            test_indices=test_df_partition.index,
        )

        return data_split
