from rdkit.Chem.rdchem import Atom

from chemtorch.components.representation.graph.featurizer.featurizer_base import FeaturizerBase


class AtomHasConjugatedBondFeaturizer(FeaturizerBase[Atom]):
    """Atom featurizer that checks if the atom is part of a conjugated system."""

    def __init__(self):
        features = [self._has_atom_conjugated_bond]
        super().__init__(features)

    def _has_atom_conjugated_bond(self, atom: Atom) -> bool:
        """Check if the atom has a conjugated bond."""
        return any(bond.GetIsConjugated() for bond in atom.GetBonds())
