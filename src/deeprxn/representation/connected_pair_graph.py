from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Tuple

from omegaconf import DictConfig
from rdkit import Chem

from deeprxn.representation.rxn_graph import AtomOriginType, RxnGraphBase


class ConnectedPairGraph(RxnGraphBase):
    """Connected pair representation with separate reactant and product graphs."""

    def __init__(
        self,
        smiles: str,
        atom_featurizer: callable,
        bond_featurizer: callable,
        connection_direction: str = "bidirectional",
        transform_cfg: Optional[DictConfig] = None,
    ):
        """Initialize connected pair graph.

        Args:
            reaction_smiles: SMARTS reaction string with atom mapping
            atom_featurizer: Function to generate atom features
            bond_featurizer: Function to generate bond features
            connection_direction: How to connect corresponding atoms:
                None: No connections
                "bidirectional": Both directions
                "reactants_to_products": Reactant to product only
                "products_to_reactants": Product to reactant only
        """
        super().__init__(
            smiles, atom_featurizer, bond_featurizer, transform_cfg
        )
        self.connection_direction = connection_direction

        # build connected pair graph
        self._build_graph()

        # apply transformations
        self._apply_transforms()

    def _build_reactant_graph(self):
        """Build graph for reactant molecules."""
        # Add reactant atom features
        for i in range(self.n_atoms):
            self.f_atoms.append(
                self.atom_featurizer(self.mol_reac.GetAtomWithIdx(i))
            )
            self.atom_origin_type.append(AtomOriginType.REACTANT)

            # Add reactant bonds
            for j in range(i + 1, self.n_atoms):
                bond = self.mol_reac.GetBondBetweenAtoms(i, j)
                if bond:
                    f_bond = self.bond_featurizer(bond)
                    self.f_bonds.extend([f_bond, f_bond])
                    self.edge_index.extend([(i, j), (j, i)])

    def _build_product_graph(self):
        """Build graph for product molecules."""
        offset = self.n_atoms  # Offset for product atom indices

        # Add product atom features
        for i in range(self.n_atoms):
            self.f_atoms.append(
                self.atom_featurizer(
                    self.mol_prod.GetAtomWithIdx(self.ri2pi[i])
                )
            )
            self.atom_origin_type.append(AtomOriginType.PRODUCT)

            # Add product bonds
            for j in range(i + 1, self.n_atoms):
                bond = self.mol_prod.GetBondBetweenAtoms(
                    self.ri2pi[i], self.ri2pi[j]
                )
                if bond:
                    f_bond = self.bond_featurizer(bond)
                    self.f_bonds.extend([f_bond, f_bond])
                    self.edge_index.extend(
                        [(i + offset, j + offset), (j + offset, i + offset)]
                    )

    def _connect_graphs(self):
        """Add edges connecting corresponding atoms in reactants and products."""
        if self.connection_direction == None:
            return

        offset = self.n_atoms

        # Zero vector for connecting edge features
        f_bond = [0] * len(self.bond_featurizer(None))

        for i in range(self.n_atoms):
            if self.connection_direction == "bidirectional":
                self.f_bonds.extend([f_bond, f_bond])
                self.edge_index.extend([(i, i + offset), (i + offset, i)])
            elif self.connection_direction == "reactants_to_products":
                self.f_bonds.append(f_bond)
                self.edge_index.append((i, i + offset))
            elif self.connection_direction == "products_to_reactants":
                self.f_bonds.append(f_bond)
                self.edge_index.append((i + offset, i))

    def _build_graph(self):
        """Build connected pair representation.

        Creates two separate graphs for reactants and products,
        then optionally connects corresponding atoms.
        """
        self._build_reactant_graph()
        self._build_product_graph()
        self._connect_graphs()
