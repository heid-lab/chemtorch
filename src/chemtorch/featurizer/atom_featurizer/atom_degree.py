from rdkit.Chem import Atom
from chemtorch.featurizer.featurizer_base import FeaturizerBase


class AtomDegreeFeaturizer(FeaturizerBase[Atom]):
    """Atom degree featurizer."""

    def __init__(self):
        features = [
            (Atom.GetTotalDegree, list(range(6))),
        ]
        super().__init__(features)
