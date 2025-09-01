from typing import List, Dict
import warnings
import pandas as pd
from collections import defaultdict

from chemtorch.utils.reaction_utils import get_reaction_core, smarts2smarts, unmap_smarts


try:
    # Python ≥ 3.12
    from typing import override  # type: ignore
except ImportError:
    # Python < 3.12
    from typing_extensions import override  # type: ignore

from chemtorch.components.data_pipeline.data_splitter.smiles_group_splitter_base import SMILESGroupSplitterBase


class ReactionCoreSplitter(SMILESGroupSplitterBase):
    def __init__(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        include_chirality: bool = False,
        save_path: str | None = None,
    ):
        """
        Initializes the ReactionCoreSplitter.

        Splits data by grouping reactions based on their reaction core/template, ensuring that
        all reactions with the same reaction core are in the same split (train, val, or test).
        This is a method to test a model's ability to generalize to new reaction types.

        Args:
            train_ratio (float): The desired ratio of data for the training set.
            val_ratio (float): The desired ratio of data for the validation set.
            test_ratio (float): The desired ratio of data for the test set.
            include_chirality (bool): If `True`, includes chirality in the reaction core generation.
            save_path (str | None, optional): If provided, saves split indices as pickle file.
        """
        super().__init__(train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, save_path=save_path)
        self.include_chirality = include_chirality

    @override
    def _make_group_id_from_smiles(self, smiles: str) -> str | None:
        """
        Generates the reaction core template for a given reaction SMILES.

        This method extracts the reaction core (the atoms that actually change during the reaction)
        and creates an unmapped reaction template that can be used to group reactions with the
        same transformation pattern.

        Args:
            smiles (str): The reaction SMILES string (e.g., 'reactant>>product').

        Returns:
            str | None: The unmapped reaction template string, or None if the reaction
                       cannot be processed or has no identifiable core.
        """
        if pd.isna(smiles) or not isinstance(smiles, str) or smiles.strip() == "":
            return None
            
        if ">>" not in smiles:
            raise ValueError(f"Invalid reaction SMILES format: {smiles}")

        r_smiles, p_smiles = smiles.split(">>")
        
        rxn_core, _ = get_reaction_core(r_smiles, p_smiles)

        # Handle empty reaction core
        if rxn_core and rxn_core.strip() == ">>":
            return None

        r_template_smiles, p_template_smiles = rxn_core.split(">>")
        
        # Unmap the reaction template to create a generic pattern
        r_unmapped_template_smiles = smarts2smarts(unmap_smarts(r_template_smiles))
        p_unmapped_template_smiles = smarts2smarts(unmap_smarts(p_template_smiles))
        
        # Create the final unmapped reaction template
        rxn_template = r_unmapped_template_smiles + ">>" + p_unmapped_template_smiles

        return rxn_template
