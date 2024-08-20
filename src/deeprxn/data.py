import os
from enum import IntEnum

import pandas as pd
import torch
import torch_geometric as tg
from rdkit import Chem
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from deeprxn.featurizer import make_featurizer


def make_mol(smi):
    params = Chem.SmilesParserParams()
    params.removeHs = False

    parts = smi.split(".")
    mols = []
    atom_origins = []
    current_atom_idx = 0

    for i, part in enumerate(parts):
        mol = Chem.MolFromSmiles(part, params)
        if mol is None:
            continue
        atom_origins.extend([i] * mol.GetNumAtoms())
        current_atom_idx += mol.GetNumAtoms()

    return Chem.MolFromSmiles(smi, params), atom_origins


def map_reac_to_prod(mol_reac, mol_prod):
    # TODO: add docstring
    prod_map_to_id = dict(
        [(atom.GetAtomMapNum(), atom.GetIdx()) for atom in mol_prod.GetAtoms()]
    )
    reac_id_to_prod_id = dict(
        [
            (atom.GetIdx(), prod_map_to_id[atom.GetAtomMapNum()])
            for atom in mol_reac.GetAtoms()
        ]
    )
    return reac_id_to_prod_id


class AtomOriginType(IntEnum):
    REACTANT = 0
    PRODUCT = 1
    DUMMY = 2


class MolGraph:
    # TODO: add docstring
    def __init__(self, smiles, atom_featurizer, bond_featurizer):
        self.smiles = smiles
        self.f_atoms = []
        self.f_bonds = []
        self.edge_index = []

        mol = make_mol(self.smiles)
        n_atoms = mol.GetNumAtoms()

        for a1 in range(n_atoms):
            f_atom = atom_featurizer(mol.GetAtomWithIdx(a1))
            self.f_atoms.append(f_atom)

            for a2 in range(a1 + 1, n_atoms):
                bond = mol.GetBondBetweenAtoms(a1, a2)
                if bond is None:
                    continue
                f_bond = bond_featurizer(bond)
                self.f_bonds.append(f_bond)
                self.f_bonds.append(f_bond)
                self.edge_index.extend([(a1, a2), (a2, a1)])


class RxnGraph:
    # TODO: add docstring
    # TODO (maybe): add support for unbalanced reactions?
    def __init__(
        self,
        smiles,
        atom_featurizer,
        bond_featurizer,
        representation="CGR",
        connection_direction="bidirectional",
        dummy_node=None,
        dummy_connection="to_dummy",
        dummy_dummy_connection="bidirectional",
    ):
        self.smiles_reac, _, self.smiles_prod = smiles.split(">")
        self.f_atoms = []
        self.f_bonds = []
        self.edge_index = []
        self.atom_origins = []

        self.mol_reac, self.reac_origins = make_mol(self.smiles_reac)
        self.mol_prod, self.prod_origins = make_mol(self.smiles_prod)
        self.ri2pi = map_reac_to_prod(self.mol_reac, self.mol_prod)
        self.n_atoms = self.mol_reac.GetNumAtoms()

        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer
        self.representation = representation
        self.connection_direction = connection_direction

        valid_connection_directions = [
            None,
            "bidirectional",
            "reactants_to_products",
            "products_to_reactants",
        ]
        if connection_direction not in valid_connection_directions:
            raise ValueError(
                f"Invalid connection_direction. Choose from: {', '.join(map(str, valid_connection_directions))}"
            )
        self.connection_direction = connection_direction

        self.is_real_bond = []
        self.atom_origin_type = []
        valid_dummy_nodes = [
            None,
            "global",
            "reactant_product",
            "all_separate",
        ]
        if dummy_node not in valid_dummy_nodes:
            raise ValueError(
                f"Invalid dummy_node. Choose from: {', '.join(map(str, valid_dummy_nodes))}"
            )
        self.dummy_node = dummy_node

        valid_dummy_connections = ["to_dummy", "from_dummy", "bidirectional"]
        if dummy_connection not in valid_dummy_connections:
            raise ValueError(
                f"Invalid dummy_connection. Choose from: {', '.join(valid_dummy_connections)}"
            )
        self.dummy_connection = dummy_connection

        self.dummy_dummy_connection = dummy_dummy_connection

        if self.representation == "CGR":
            self._build_cgr()
        elif representation == "connected_pair":
            self._build_connected_pair()
        elif representation == "fully_connected":
            self._build_fully_connected()
        else:
            raise ValueError(
                "Invalid representation. Choose 'CGR', 'fully_connected', 'connected_pair'"
            )

    def _get_atom_features(self, i):
        f_atom_reac = self.atom_featurizer(self.mol_reac.GetAtomWithIdx(i))
        f_atom_prod = self.atom_featurizer(
            self.mol_prod.GetAtomWithIdx(self.ri2pi[i])
        )
        f_atom_diff = [y - x for x, y in zip(f_atom_reac, f_atom_prod)]
        return f_atom_reac + f_atom_diff

    def _get_bond_features(self, bond_reac, bond_prod):
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

    def _build_fully_connected(self):
        # Build reactant graph
        for i in range(self.n_atoms):
            self.f_atoms.append(
                self.atom_featurizer(self.mol_reac.GetAtomWithIdx(i))
            )
            self.atom_origin_type.append(AtomOriginType.REACTANT)

        for i in range(self.n_atoms):
            for j in range(i + 1, self.n_atoms):
                bond_reac = self.mol_reac.GetBondBetweenAtoms(i, j)
                f_bond = (
                    self.bond_featurizer(bond_reac)
                    if bond_reac
                    else [0] * len(self.bond_featurizer(None))
                )
                self.f_bonds.extend([f_bond, f_bond])
                self.edge_index.extend([(i, j), (j, i)])
                self.is_real_bond.extend(
                    [bond_reac is not None, bond_reac is not None]
                )

        # Build product graph
        offset = self.n_atoms
        for i in range(self.n_atoms):
            self.f_atoms.append(
                self.atom_featurizer(
                    self.mol_prod.GetAtomWithIdx(self.ri2pi[i])
                )
            )
            self.atom_origin_type.append(AtomOriginType.PRODUCT)

        for i in range(self.n_atoms):
            for j in range(i + 1, self.n_atoms):
                bond_prod = self.mol_prod.GetBondBetweenAtoms(
                    self.ri2pi[i], self.ri2pi[j]
                )
                f_bond = (
                    self.bond_featurizer(bond_prod)
                    if bond_prod
                    else [0] * len(self.bond_featurizer(None))
                )
                self.f_bonds.extend([f_bond, f_bond])
                self.edge_index.extend(
                    [(i + offset, j + offset), (j + offset, i + offset)]
                )
                self.is_real_bond.extend(
                    [bond_prod is not None, bond_prod is not None]
                )

        # Connect all nodes across reactants and products based on connection_direction
        f_bond = [0] * len(self.bond_featurizer(None))
        if self.connection_direction is not None:
            for i in range(self.n_atoms):
                if self.connection_direction == "bidirectional":
                    self.f_bonds.extend([f_bond, f_bond])
                    self.edge_index.extend([(i, i + offset), (i + offset, i)])
                    self.is_real_bond.extend([False, False])
                elif self.connection_direction == "reactants_to_products":
                    self.f_bonds.append(f_bond)
                    self.edge_index.append((i, i + offset))
                    self.is_real_bond.append(False)
                elif self.connection_direction == "products_to_reactants":
                    self.f_bonds.append(f_bond)
                    self.edge_index.append((i + offset, i))
                    self.is_real_bond.append(False)

        # Add dummy nodes if specified
        if self.dummy_node:
            dummy_feature = torch.ones(
                len(self.atom_featurizer(self.mol_reac.GetAtomWithIdx(0)))
            )
            f_bond = [0] * len(self.bond_featurizer(None))

            if self.dummy_node == "global":
                self._add_global_dummy(dummy_feature, f_bond)
            elif self.dummy_node == "reactant_product":
                self._add_reactant_product_dummies(dummy_feature, f_bond)
            elif self.dummy_node == "all_separate":
                self._add_all_separate_dummies(dummy_feature, f_bond)

    def _build_cgr(self):
        for i in range(self.n_atoms):
            self.f_atoms.append(self._get_atom_features(i))

            for j in range(i + 1, self.n_atoms):
                bond_reac = self.mol_reac.GetBondBetweenAtoms(i, j)
                bond_prod = self.mol_prod.GetBondBetweenAtoms(
                    self.ri2pi[i], self.ri2pi[j]
                )
                if bond_reac is None and bond_prod is None:
                    continue
                f_bond = self._get_bond_features(bond_reac, bond_prod)
                self.f_bonds.append(f_bond)
                self.f_bonds.append(f_bond)
                self.edge_index.extend([(i, j), (j, i)])

    def _build_connected_pair(self):
        # Build reactant graph
        for i in range(self.n_atoms):
            self.f_atoms.append(
                self.atom_featurizer(self.mol_reac.GetAtomWithIdx(i))
            )
            self.atom_origin_type.append(AtomOriginType.REACTANT)
            self.atom_origins.append(self.reac_origins[i])

            for j in range(i + 1, self.n_atoms):
                bond_reac = self.mol_reac.GetBondBetweenAtoms(i, j)
                if bond_reac:
                    f_bond = self.bond_featurizer(bond_reac)
                    self.f_bonds.append(f_bond)
                    self.f_bonds.append(f_bond)
                    self.edge_index.extend([(i, j), (j, i)])
                    self.is_real_bond.extend([True, True])

        # Build product graph
        offset = self.n_atoms
        for i in range(self.n_atoms):
            self.f_atoms.append(
                self.atom_featurizer(
                    self.mol_prod.GetAtomWithIdx(self.ri2pi[i])
                )
            )
            self.atom_origin_type.append(AtomOriginType.PRODUCT)
            self.atom_origins.append(
                self.prod_origins[self.ri2pi[i]] + max(self.reac_origins) + 1
            )

            for j in range(i + 1, self.n_atoms):
                bond_prod = self.mol_prod.GetBondBetweenAtoms(
                    self.ri2pi[i], self.ri2pi[j]
                )
                if bond_prod:
                    f_bond = self.bond_featurizer(bond_prod)
                    self.f_bonds.append(f_bond)
                    self.f_bonds.append(f_bond)
                    self.edge_index.extend(
                        [(i + offset, j + offset), (j + offset, i + offset)]
                    )
                    self.is_real_bond.extend([True, True])

        # Connect corresponding atoms between reactants and products
        if self.connection_direction is not None:
            for i in range(self.n_atoms):
                f_bond = [0] * len(
                    self.bond_featurizer(None)
                )  # Use a zero vector for the connecting edge
                if self.connection_direction == "bidirectional":
                    self.f_bonds.append(f_bond)
                    self.f_bonds.append(f_bond)
                    self.edge_index.extend([(i, i + offset), (i + offset, i)])
                    self.is_real_bond.extend([False, False])
                elif self.connection_direction == "reactants_to_products":
                    self.f_bonds.append(f_bond)
                    self.edge_index.append((i, i + offset))
                    self.is_real_bond.append(False)
                elif self.connection_direction == "products_to_reactants":
                    self.f_bonds.append(f_bond)
                    self.edge_index.append((i + offset, i))
                    self.is_real_bond.append(False)

        # Add dummy nodes if specified
        if self.dummy_node:
            dummy_feature = torch.zeros(
                len(self.atom_featurizer(self.mol_reac.GetAtomWithIdx(0)))
            )
            f_bond = [0] * len(self.bond_featurizer(None))

            if self.dummy_node == "global":
                self._add_global_dummy(dummy_feature, f_bond)
            elif self.dummy_node == "reactant_product":
                self._add_reactant_product_dummies(dummy_feature, f_bond)
            elif self.dummy_node == "all_separate":
                self._add_all_separate_dummies(dummy_feature, f_bond)

    def _connect_dummy_to_node(self, dummy_idx, node_idx, f_bond):
        if self.dummy_connection in ["bidirectional", "from_dummy"]:
            self.f_bonds.append(f_bond)
            self.edge_index.append((dummy_idx, node_idx))
            self.is_real_bond.append(False)
        if self.dummy_connection in ["bidirectional", "to_dummy"]:
            self.f_bonds.append(f_bond)
            self.edge_index.append((node_idx, dummy_idx))
            self.is_real_bond.append(False)

    def _add_global_dummy(self, dummy_feature, f_bond):
        dummy_idx = len(self.f_atoms)
        self.f_atoms.append(dummy_feature.tolist())
        self.atom_origins.append(-1)  # Use -1 for dummy nodes
        self.atom_origin_type.append(AtomOriginType.DUMMY)

        for i in range(2 * self.n_atoms):
            self._connect_dummy_to_node(dummy_idx, i, f_bond)

    def _add_reactant_product_dummies(self, dummy_feature, f_bond):
        dummy_reactant_idx = len(self.f_atoms)
        self.f_atoms.append(dummy_feature.tolist())
        self.atom_origins.append(-1)
        self.atom_origin_type.append(AtomOriginType.DUMMY)

        dummy_product_idx = len(self.f_atoms)
        self.f_atoms.append(dummy_feature.tolist())
        self.atom_origins.append(-1)
        self.atom_origin_type.append(AtomOriginType.DUMMY)

        for i in range(self.n_atoms):
            self._connect_dummy_to_node(dummy_reactant_idx, i, f_bond)
            self._connect_dummy_to_node(
                dummy_product_idx, i + self.n_atoms, f_bond
            )

        if self.dummy_dummy_connection == "bidirectional":
            self.f_bonds.extend([f_bond, f_bond])
            self.edge_index.extend(
                [
                    (dummy_reactant_idx, dummy_product_idx),
                    (dummy_product_idx, dummy_reactant_idx),
                ]
            )
            self.is_real_bond.extend([False, False])

    def _add_all_separate_dummies(self, dummy_feature, f_bond):
        unique_origins = set(self.atom_origins)
        dummy_indices = {}

        for origin in unique_origins:
            dummy_idx = len(self.f_atoms)
            self.f_atoms.append(dummy_feature.tolist())
            self.atom_origins.append(-1)
            self.atom_origin_type.append(AtomOriginType.DUMMY)
            dummy_indices[origin] = dummy_idx

        for i, origin in enumerate(self.atom_origins):
            if origin in dummy_indices:
                self._connect_dummy_to_node(dummy_indices[origin], i, f_bond)

        if self.dummy_dummy_connection == "bidirectional":
            dummy_list = list(dummy_indices.values())
            for i in range(len(dummy_list)):
                for j in range(i + 1, len(dummy_list)):
                    self.f_bonds.extend([f_bond, f_bond])
                    self.edge_index.extend(
                        [
                            (dummy_list[i], dummy_list[j]),
                            (dummy_list[j], dummy_list[i]),
                        ]
                    )
                    self.is_real_bond.extend([False, False])


class ChemDataset(Dataset):
    # TODO: add docstring
    # TODO: add functionality to drop invalid molecules
    # TODO: add option to cache graphs
    def __init__(
        self,
        smiles,
        labels,
        atom_featurizer,
        bond_featurizer,
        mode="mol",
        representation="CGR",
        connection_direction="bidirectional",
        dummy_node=None,
        dummy_connection="to_dummy",
        dummy_dummy_connection="bidirectional",
    ):

        super(ChemDataset, self).__init__()
        self.smiles = smiles
        self.labels = labels
        self.mode = mode
        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer
        self.representation = representation
        self.connection_direction = connection_direction
        self.dummy_node = dummy_node
        self.dummy_connection = dummy_connection
        self.dummy_dummy_connection = dummy_dummy_connection

    def process_key(self, key):
        # TODO: add docstring
        smi = self.smiles[key]
        if self.mode == "mol":
            molgraph = MolGraph(
                smi, self.atom_featurizer, self.bond_featurizer
            )
        elif self.mode == "rxn":
            molgraph = RxnGraph(
                smi,
                self.atom_featurizer,
                self.bond_featurizer,
                self.representation,
                self.connection_direction,
                self.dummy_node,
                self.dummy_connection,
                self.dummy_dummy_connection,
            )
        else:
            raise ValueError("Unknown option for mode", self.mode)
        mol = self.molgraph2data(molgraph, key)
        return mol

    def molgraph2data(self, molgraph, key):
        data = tg.data.Data()
        data.x = torch.tensor(molgraph.f_atoms, dtype=torch.float)
        data.edge_index = (
            torch.tensor(molgraph.edge_index, dtype=torch.long)
            .t()
            .contiguous()
        )
        data.edge_attr = torch.tensor(molgraph.f_bonds, dtype=torch.float)
        data.y = torch.tensor([self.labels[key]], dtype=torch.float)
        data.smiles = self.smiles[key]
        data.is_real_bond = torch.tensor(
            molgraph.is_real_bond, dtype=torch.bool
        )
        data.atom_origin_type = torch.tensor(
            molgraph.atom_origin_type, dtype=torch.long
        )
        data.atom_origins = torch.tensor(
            molgraph.atom_origins, dtype=torch.long
        )
        return data

    def get(self, key):
        # TODO: add docstring
        return self.process_key(key)

    def len(self):
        # TODO: add docstring
        return len(self.smiles)


def get_data_paths(data_folder):
    base_path = os.path.join("data", data_folder)
    return {
        "train": os.path.join(base_path, "train.csv"),
        "val": os.path.join(base_path, "val.csv"),
        "test": os.path.join(base_path, "test.csv"),
    }


def load_from_csv(dataset_name, split):
    # TODO: add docstring

    data_path = get_data_paths(dataset_name)

    # barriers_cycloadd, barriers_e2, barriers_rdb7, barriers_rgd1 ,barriers_sn2
    if dataset_name == "barriers_cycloadd":
        input_column = "rxn_smiles"
        target_column = "G_act"
    elif dataset_name == "barriers_e2":
        input_column = "AAM"
        target_column = "ea"
    elif dataset_name == "barriers_rdb7" or dataset_name == "barriers_rgd1":
        input_column = "smiles"
        target_column = "ea"
    elif dataset_name == "barriers_sn2":
        input_column = "AAM"
        target_column = "ea"
    else:
        raise ValueError("Unknown dataset", dataset_name)

    data_df = pd.read_csv(data_path[split])
    smiles = data_df[input_column].values
    labels = data_df[target_column].values.astype(float)
    return smiles, labels


def construct_loader(
    smiles,
    labels,
    atom_featurizer,
    bond_featurizer,
    num_workers,
    shuffle=True,
    batch_size=50,
    mode="rxn",
    representation="CGR",
    connection_direction="products_to_reactants",
    dummy_node=None,
    dummy_connection="to_dummy",
    dummy_dummy_connection="bidirectional",
):
    # TODO: add docstring
    dataset = ChemDataset(
        smiles,
        labels,
        atom_featurizer,
        bond_featurizer,
        mode,
        representation,
        connection_direction,
        dummy_node,
        dummy_connection,
        dummy_dummy_connection,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        sampler=None,
        generator=torch.Generator().manual_seed(0),
    )
    return loader


class Standardizer:
    # TODO: add docstring
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x, rev=False):
        if rev:
            return (x * self.std) + self.mean
        return (x - self.mean) / self.std
