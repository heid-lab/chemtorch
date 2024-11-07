from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import hydra
from omegaconf import DictConfig
from rdkit import Chem

from deeprxn.representation.rxn_graph import AtomOriginType, RxnGraphBase


class CGRGraph(RxnGraphBase):
    """Condensed Graph of Reaction (CGR) representation."""

    def __init__(
        self,
        smiles: str,
        atom_featurizer: callable,
        bond_featurizer: callable,
        transform_cfg: Optional[DictConfig] = None,
    ):
        """Initialize CGR graph.

        Args:
            reaction_smiles: reaction string with atom mapping
            atom_featurizer: Function to generate atom features
            bond_featurizer: Function to generate bond features
        """
        super().__init__(
            smiles, atom_featurizer, bond_featurizer, transform_cfg
        )

        # build CGR graph
        self._build_graph()

        # apply transformations
        self._apply_transforms()

    def _get_atom_features(self, atom_idx: int) -> List[float]:
        """Generate features for an atom in CGR representation.

        Concatenates reactant features with feature difference (prod - reac).

        Args:
            atom_idx: Index of atom in reactant molecule

        Returns:
            Combined feature vector [reac_feat, feat_diff]
        """
        f_atom_reac = self.atom_featurizer(
            self.mol_reac.GetAtomWithIdx(atom_idx)
        )
        f_atom_prod = self.atom_featurizer(
            self.mol_prod.GetAtomWithIdx(self.ri2pi[atom_idx])
        )
        f_atom_diff = [y - x for x, y in zip(f_atom_reac, f_atom_prod)]
        return f_atom_reac + f_atom_diff

    def _get_bond_features(
        self, bond_reac: Chem.Bond, bond_prod: Chem.Bond
    ) -> List[float]:
        """Generate features for a bond in CGR representation.

        Concatenates reactant features with feature difference (prod - reac).

        Args:
            bond_reac: Bond in reactant molecule (or None)
            bond_prod: Bond in product molecule (or None)

        Returns:
            Combined feature vector [reac_feat, feat_diff]
        """
        f_bond_reac = (
            self.bond_featurizer(bond_reac)
            if bond_reac
            else [0] * len(self.bond_featurizer(None))
        )
        f_bond_prod = (
            self.bond_featurizer(bond_prod)
            if bond_prod
            else [0] * len(self.bond_featurizer(None))
        )
        f_bond_diff = [y - x for x, y in zip(f_bond_reac, f_bond_prod)]
        return f_bond_reac + f_bond_diff

    def _build_graph(self):
        """Build CGR representation.

        Creates a single graph where:
        - Each atom appears once
        - Atom features combine reactant state and change
        - Bond features combine reactant state and change
        """
        n_atoms = self.mol_reac.GetNumAtoms()

        for i in range(n_atoms):
            self.f_atoms.append(self._get_atom_features(i))
            self.atom_origin_type.append(AtomOriginType.REACTANT_PRODUCT)

            for j in range(i + 1, n_atoms):
                bond_reac = self.mol_reac.GetBondBetweenAtoms(i, j)
                bond_prod = self.mol_prod.GetBondBetweenAtoms(
                    self.ri2pi[i], self.ri2pi[j]
                )

                # only add if bond exists in either reactants or products
                if bond_reac is None and bond_prod is None:
                    continue

                f_bond = self._get_bond_features(bond_reac, bond_prod)
                # add bond in both directions (i->j and j->i)
                self.f_bonds.extend([f_bond, f_bond])
                self.edge_index.extend([(i, j), (j, i)])
