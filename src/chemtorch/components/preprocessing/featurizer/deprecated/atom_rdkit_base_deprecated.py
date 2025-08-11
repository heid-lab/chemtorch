from rdkit.Chem.rdchem import Atom, HybridizationType

from chemtorch.components.preprocessing.featurizer.featurizer_base import FeaturizerBase


class RDKitBaseAtomFeaturizer(FeaturizerBase[Atom]):
    """Atom featurizer with organic chemistry features."""

    def __init__(self):
        features = [
            (Atom.GetAtomicNum, list(range(1, 37)) + [53]),
            (Atom.GetTotalDegree, list(range(6))),
            (Atom.GetFormalCharge, [-2, -1, 0, 1, 2]),
            (Atom.GetTotalNumHs, list(range(5))),
            (
                Atom.GetHybridization,
                [
                    HybridizationType.S,
                    HybridizationType.SP,
                    HybridizationType.SP2,
                    HybridizationType.SP2D,
                    HybridizationType.SP3,
                    HybridizationType.SP3D,
                    HybridizationType.SP3D2,
                ],
            ),
            Atom.GetIsAromatic,
            lambda atom: Atom.GetMass(atom) * 0.01,
        ]
        super().__init__(features)
