from rdkit.Chem import Atom

from chemtorch.components.preprocessing.featurizer.featurizer_base import FeaturizerBase


class AtomIsAromaticFeaturizer(FeaturizerBase[Atom]):
    """Atom is aromatic featurizer."""

    def __init__(self):
        features = [Atom.GetIsAromatic]
        super().__init__(features)
