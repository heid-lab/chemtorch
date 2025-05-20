from rdkit.Chem.rdchem import Atom, HybridizationType

from deepreaction.featurizer.featurizer_base import FeaturizerBase


class RDKitOrganicAtomFeaturizer(FeaturizerBase[Atom]):
    """Atom featurizer with organic chemistry features."""
    
    def __init__(self):
        features = [
            (Atom.GetAtomicNum, [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]),
            (Atom.GetTotalDegree, list(range(6))),
            (Atom.GetFormalCharge, [-2, -1, 0, 1, 2]),
            (Atom.GetTotalNumHs, list(range(5))),
            (Atom.GetHybridization, [
                HybridizationType.S, 
                HybridizationType.SP, 
                HybridizationType.SP2, 
                HybridizationType.SP3
            ]),
            Atom.GetIsAromatic,
            lambda item: Atom.GetMass(item) * 0.01,
            lambda item: Atom.IsInRingSize(item, 3),
            lambda item: Atom.IsInRingSize(item, 4),
            lambda item: Atom.IsInRingSize(item, 5),
            lambda item: Atom.IsInRingSize(item, 6),
            lambda item: Atom.IsInRingSize(item, 7),
        ]
        super().__init__(features)