from rdkit.Chem.rdchem import Atom

from chemtorch.components.representation.graph.featurizer.featurizer_base import FeaturizerBase


class AtomicNumberFeaturizer(FeaturizerBase[Atom]):
    """Atom featurizer using only atomic number."""

    def __init__(self) -> None:
        features = [Atom.GetAtomicNum]
        super().__init__(features)


class OrganicAtomicNumberOneHotFeaturizer(FeaturizerBase[Atom]):
    """Atom featurizer using only atomic number."""

    def __init__(self):
        features = [
            (Atom.GetAtomicNum, [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]),
        ]
        super().__init__(features)
