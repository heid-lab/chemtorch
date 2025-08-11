from rdkit.Chem import Atom

from chemtorch.components.preprocessing.featurizer.featurizer_base import FeaturizerBase


class OrganicAtomicNumberOneHotFeaturizer(FeaturizerBase[Atom]):
    """Atom featurizer using only atomic number."""

    def __init__(self):
        features = [(Atom.GetAtomicNum, list(range(1, 37)) + [53])]
        super().__init__(features)
