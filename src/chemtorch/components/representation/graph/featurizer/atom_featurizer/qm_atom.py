from typing import List, Dict, Callable
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Atom

from chemtorch.components.representation.graph.featurizer.featurizer_base import FeaturizerBase


class QMAtomFeaturizer(FeaturizerBase[Atom]):
    """Atom featurizer using external qm features."""

    def __init__(self, path: str):
        """
        Args:
            path (str): Path to the pickle file containing the features.
        """
        orig = pd.read_pickle(f"{path}")
        features_dict = self._restructure_features_dict(orig)
        features = self._make_qm_feature_fns(features_dict)
        super().__init__(features)

    @staticmethod
    def _restructure_features_dict(orig: Dict) -> Dict[int, Dict[str, List[float]]]:
        """
        Restructure dict from smiles->feature_idx->atom_idx to feature_idx->smiles->atom_idx.
        """
        features_dict = {}
        for smiles, feature_list in orig.items():
            for i, arr in enumerate(feature_list):
                if i not in features_dict:
                    features_dict[i] = {}
                features_dict[i][smiles] = arr
        return features_dict

    @staticmethod
    def _make_qm_feature_fns(
        features_dict: Dict[int, Dict[str, List[float]]],
    ) -> List[Callable[[Atom], float]]:
        """
        Create a list of QM atom feature functions.
        Each function takes an atom and returns the corresponding feature value extracted from the features_dict.

        Args:
            features_dict (Dict[int, Dict[str, List[float]]]): Feature dictionary structured as
                feature_idx -> smiles -> atom_idx.
        Returns:
            List[Callable[[Atom], float]]: List of functions that return QM features for molecule atoms.
        """

        def get_mol_smiles(atom: Atom) -> str:
            return Chem.MolToSmiles(atom.GetOwningMol())

        def make_qm_feature_fn(qm_feature_idx: int):
            def qm_feature_fn(atom: Atom):
                smiles = get_mol_smiles(atom)
                return features_dict[qm_feature_idx][smiles][atom.GetIdx()]

            return qm_feature_fn

        return [make_qm_feature_fn(i) for i in sorted(features_dict.keys())]
