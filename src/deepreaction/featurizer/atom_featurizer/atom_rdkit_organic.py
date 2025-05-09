from typing import List, Optional

from rdkit.Chem.rdchem import Atom, HybridizationType

from deepreaction.featurizer.atom_featurizer.featurizer_base import FeaturizerBase


class RDKitOrganicAtomFeaturizer(FeaturizerBase):
    """Atom featurizer with organic chemistry features."""
    
    def __init__(self):
        self.features = [
            (Atom.GetAtomicNum, [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]),
            (Atom.GetTotalDegree, list(range(6))),
            (Atom.GetFormalCharge, [-2, -1, 0, 1, 2]),
            (Atom.GetTotalNumHs, list(range(5))),
            (Atom.GetHybridization, [
                HybridizationType.S, HybridizationType.SP, 
                HybridizationType.SP2, HybridizationType.SP3
            ]),
            (lambda item: Atom.GetIsAromatic(item), []),
            (lambda item: Atom.GetMass(item) * 0.01, []),
            (lambda item: Atom.IsInRingSize(item, 3), []),
            (lambda item: Atom.IsInRingSize(item, 4), []),
            (lambda item: Atom.IsInRingSize(item, 5), []),
            (lambda item: Atom.IsInRingSize(item, 6), []),
            (lambda item: Atom.IsInRingSize(item, 7), []),
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
