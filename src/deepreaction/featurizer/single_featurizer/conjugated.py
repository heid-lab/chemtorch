from typing import List, Optional

from rdkit.Chem.rdchem import Atom

from deepreaction.featurizer.single_featurizer.featurizer_base import FeaturizerBase


class ConjugatedFeaturizer(FeaturizerBase):
    """Atom featurizer that checks if the atom is part of a conjugated system."""
    
    def __init__(self):
        pass
    
    def __call__(self, atom: Optional[Atom]) -> int:
        if atom is None:
            return 0
        
        # Check if any bonds to this atom are conjugated
        bonds = atom.GetBonds()
        for bond in bonds:
            if bond.GetIsConjugated():
                return 1

        return 0
