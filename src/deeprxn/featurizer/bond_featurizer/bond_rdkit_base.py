from typing import List, Optional

from rdkit.Chem.rdchem import Bond, BondType

from deeprxn.featurizer.bond_featurizer.featurizer_base import FeaturizerBase


class RDKitBaseBondFeaturizer(FeaturizerBase):
    """Bond featurizer with standard RDKit features."""
    
    def __init__(self):
        self.features = [
            (Bond.GetBondType, [
                BondType.SINGLE, BondType.DOUBLE, 
                BondType.TRIPLE, BondType.AROMATIC,
            ]),
            (lambda item: Bond.GetIsConjugated(item), []),
            (lambda item: Bond.IsInRingSize(item, 3), []),
            (lambda item: Bond.IsInRingSize(item, 4), []),
            (lambda item: Bond.IsInRingSize(item, 5), []),
            (lambda item: Bond.IsInRingSize(item, 6), []),
            (lambda item: Bond.IsInRingSize(item, 7), []),
        ]
    
    def __call__(self, bond: Optional[Bond]) -> List[float]:
        if bond is None:
            # calculate null feature size
            dim = sum([(len(options) + 1) if options else 1 for _, options in self.features])
            return [0] * dim
        
        features = []
        for func, options in self.features:
            if options:
                features.extend(self._one_hot_unk(bond, func, options))
            else:
                features.append(func(bond))
        
        return features
    
    def _one_hot_unk(self, item, func, options):
        x = [0] * (len(options) + 1)
        option_dict = {j: i for i, j in enumerate(options)}
        x[option_dict.get(func(item), len(option_dict))] = 1
        return x
