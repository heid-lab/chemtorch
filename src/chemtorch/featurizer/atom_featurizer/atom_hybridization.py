from rdkit.Chem import Atom, HybridizationType

from chemtorch.featurizer.featurizer_base import FeaturizerBase


class AtomHybridizationFeaturizer(FeaturizerBase[Atom]):
    """Atom hybridization featurizer."""

    def __init__(self):
        features = [
            (
                Atom.GetHybridization,
                [
                    HybridizationType.S,
                    HybridizationType.SP,
                    HybridizationType.SP2,
                    HybridizationType.SP3,
                ],
            ),
        ]
        super().__init__(features)
