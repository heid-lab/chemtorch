from typing import List, Optional

from rdkit.Chem.rdchem import Atom

from deeprxn.featurizer.single_featurizer.featurizer_base import FeaturizerBase


class AtomicNumberFeaturizer(FeaturizerBase):
    """Atom featurizer using only atomic number."""
    
    def __init__(self):
        pass
    
    def __call__(self, atom: Optional[Atom]) -> int:
        if atom is None:
            return 0
        
        feature = atom.GetAtomicNum()

        return feature
