import pandas as pd
from rdkit import Chem


class QMAtomFeaturizer():
    """Atom featurizer using external qm features."""
    
    def __init__(self, path):
        self.features = pd.read_pickle(f"{path}")

    def __call__(self, atom):
        smiles = Chem.MolToSmiles(atom.GetOwningMol())
        features = [
            (self.features[smiles][i][atom.GetIdx()])
            for i in range(len(self.features[smiles]))
        ]

        return features


