from typing import List, Sequence

from deepreaction.featurizer.abstract_featurizer import AbstractFeaturizer

class FeaturizerCompose(AbstractFeaturizer):
    """
    Compose multiple featurizers. Each featurizer is called on the input,
    and their outputs (feature lists) are concatenated.
    """

    def __init__(self, featurizers: Sequence[AbstractFeaturizer]):
        """
        Initialize the featurizer composer with a list of featurizers.
        
        Args:
            featurizers (Sequence[FeaturizerBase]): A sequence of featurizers to be composed.

        Raises:
            ValueError: If the featurizers list is empty.
            ValueError: If any of the featurizers is not an instance of FeaturizerBase.
        """
        if not featurizers:
            raise ValueError("Featurizers list cannot be empty.")
        if not all(isinstance(featurizer, AbstractFeaturizer) for featurizer in featurizers):
            raise ValueError("All featurizers must be instances of FeaturizerBase.")
        self.featurizers = featurizers

    def __call__(self, *args, **kwargs) -> List[float]:
        features = []
        for featurizer in self.featurizers:
            features.extend(featurizer(*args, **kwargs))
        return features