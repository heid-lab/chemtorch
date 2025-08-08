from rdkit.Chem import Atom

from chemtorch.components.preprocessing.featurizer.featurizer_base import FeaturizerBase


class AtomIsInRingFeaturizer(FeaturizerBase[Atom]):
    """Atom featurizer with ring features."""

    def __init__(self):
        features = [
            lambda item: Atom.IsInRingSize(item, 3),
            lambda item: Atom.IsInRingSize(item, 4),
            lambda item: Atom.IsInRingSize(item, 5),
            lambda item: Atom.IsInRingSize(item, 6),
            lambda item: Atom.IsInRingSize(item, 7),
        ]
        super().__init__(features)
