from rdkit.Chem.rdchem import Bond, BondType

from chemtorch.components.representation.graph.featurizer.featurizer_base import FeaturizerBase


class BondTypeFeaturizer(FeaturizerBase):
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
        ]
        super().__init__(features)
