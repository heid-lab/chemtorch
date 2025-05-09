from typing import List, Optional

from rdkit.Chem.rdchem import Atom

from deepreaction.featurizer.atom_featurizer.featurizer_base import FeaturizerBase


class AtomicNumberFeaturizer(FeaturizerBase):
    """Atom featurizer using only atomic number."""
    
    def __init__(self):
        self.features = [
            (Atom.GetAtomicNum, list(range(1, 37)) + [53]),
        ]
    
    def __call__(self, atom: Optional[Atom]) -> List[float]:
        if atom is None:
            # calculate null feature size
            dim = sum([(len(options) + 1) if options else 1 for _, options in self.features])
            return [0] * dim
        
        features = []
        for func, options in self.features:
            if options:
                features.extend(self._one_hot_unk(atom, func, options))
            else:
                features.append(func(atom))
        
        return features
    
    def _one_hot_unk(self, item, func, options):
        x = [0] * (len(options) + 1)
        option_dict = {j: i for i, j in enumerate(options)}
        x[option_dict.get(func(item), len(option_dict))] = 1
        return x

