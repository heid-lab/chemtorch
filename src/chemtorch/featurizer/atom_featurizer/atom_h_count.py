from rdkit.Chem import Atom

from chemtorch.featurizer.featurizer_base import FeaturizerBase


class AtomHCountFeaturizer(FeaturizerBase[Atom]):
    """Atom H count featurizer."""

    def __init__(self):
        features = [
            (Atom.GetTotalNumHs, list(range(5))),
        ]
        super().__init__(features)
