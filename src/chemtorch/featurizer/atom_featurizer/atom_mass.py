from rdkit.Chem import Atom

from chemtorch.featurizer.featurizer_base import FeaturizerBase


class CentiAtomMassFeaturizer(FeaturizerBase[Atom]):
    """Centis atom mass featurizer."""

    def __init__(self):
        features = [lambda atom: Atom.GetMass(atom) * 0.01]
        super().__init__(features)
