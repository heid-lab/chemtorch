from typing import List, Tuple

from rdkit import Chem

################################################################################################################################
########################################## OLD, NOT USED #######################################################################
################################################################################################################################


class MolGraph:
    """Single molecule graph representation."""

    def __init__(
        self, smiles: str, atom_featurizer: callable, bond_featurizer: callable
    ):
        """Initialize molecule graph.

        Args:
            smiles: SMILES string of molecule
            atom_featurizer: Function to generate atom features
            bond_featurizer: Function to generate bond features
        """
        self.smiles = smiles
        self.f_atoms: List = []
        self.f_bonds: List = []
        self.edge_index: List[Tuple[int, int]] = []

        mol = self._make_mol(self.smiles)
        n_atoms = mol.GetNumAtoms()

        for a1 in range(n_atoms):
            f_atom = atom_featurizer(mol.GetAtomWithIdx(a1))
            self.f_atoms.append(f_atom)

            for a2 in range(a1 + 1, n_atoms):
                bond = mol.GetBondBetweenAtoms(a1, a2)
                if bond is None:
                    continue
                f_bond = bond_featurizer(bond)
                self.f_bonds.extend([f_bond, f_bond])
                self.edge_index.extend([(a1, a2), (a2, a1)])

    @staticmethod
    def _make_mol(smi: str) -> Chem.Mol:
        """Create RDKit mol object.

        Args:
            smi: SMILES string

        Returns:
            RDKit molecule object
        """
        params = Chem.SmilesParserParams()
        params.removeHs = False
        return Chem.MolFromSmiles(smi, params)
