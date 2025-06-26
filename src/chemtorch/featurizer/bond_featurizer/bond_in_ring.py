from rdkit.Chem.rdchem import Bond

from chemtorch.featurizer.featurizer_base import FeaturizerBase


class BondInRingFeaturizer(FeaturizerBase):
    """Bond featurizer with standard RDKit features."""

    def __init__(self):
        features = [
            lambda item: Bond.IsInRingSize(item, 3),
            lambda item: Bond.IsInRingSize(item, 4),
            lambda item: Bond.IsInRingSize(item, 5),
            lambda item: Bond.IsInRingSize(item, 6),
            lambda item: Bond.IsInRingSize(item, 7),
        ]
        super().__init__(features)
