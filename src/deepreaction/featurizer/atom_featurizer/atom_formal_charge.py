from rdkit.Chem import Atom

from deepreaction.featurizer.featurizer_base import FeaturizerBase


class AtomFormalChargeFeaturizer(FeaturizerBase[Atom]):
    """Atom formal charge featurizer."""

    def __init__(self):
        features = [
            (Atom.GetFormalCharge, [-2, -1, 0, 1, 2]),
        ]
        super().__init__(features)