from rdkit.Chem.rdchem import Bond, BondType

from chemtorch.components.preprocessing.featurizer.featurizer_base import FeaturizerBase


class RDKitBaseBondFeaturizer(FeaturizerBase[Bond]):
    """Bond featurizer with standard RDKit features."""

    def __init__(self):
        features = [
            (
                Bond.GetBondType,
                [
                    BondType.SINGLE,
                    BondType.DOUBLE,
                    BondType.TRIPLE,
                    BondType.AROMATIC,
                ],
            ),
            lambda item: Bond.GetIsConjugated(item),
            lambda item: Bond.IsInRingSize(item, 3),
            lambda item: Bond.IsInRingSize(item, 4),
            lambda item: Bond.IsInRingSize(item, 5),
            lambda item: Bond.IsInRingSize(item, 6),
            lambda item: Bond.IsInRingSize(item, 7),
        ]
        super().__init__(features)
