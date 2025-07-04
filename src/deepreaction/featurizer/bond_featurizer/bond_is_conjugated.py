from rdkit.Chem import Bond

from deepreaction.featurizer.featurizer_base import FeaturizerBase

class BondIsConjugatedFeaturizer(FeaturizerBase[Bond]):
    """Bond is conjugated featurizer."""

    def __init__(self):
        features = [Bond.GetIsConjugated]
        super().__init__(features)