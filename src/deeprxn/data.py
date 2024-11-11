from enum import IntEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
import torch_geometric as tg
from rdkit import Chem
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from deeprxn.featurizer.featurizer import make_featurizer
from deeprxn.utils import load_csv_dataset


class RxnGraph:
    # TODO: add docstring
    # TODO (maybe): add support for unbalanced reactions?
    # TODO: lots of improvements can be made here
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
        dummy_feat_init="zeros",
    ):
        self.smiles_reac, _, self.smiles_prod = smiles.split(">")
        self.f_atoms = []
        self.f_bonds = []
        self.edge_index = []
        self.atom_origins = []
        self.is_real_bond = []
        self.atom_origin_type = []
        incoming_edges_list = []
        incoming_edges_batch = []
        incoming_edges_batch_from_zero = []

        self.mol_reac, self.reac_origins = make_mol(self.smiles_reac)
        self.mol_prod, self.prod_origins = make_mol(self.smiles_prod)
        self.ri2pi = map_reac_to_prod(self.mol_reac, self.mol_prod)
        self.n_atoms = self.mol_reac.GetNumAtoms()

        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer
        self.representation = representation
        self.connection_direction = connection_direction
        self.dummy_feat_init = dummy_feat_init

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
        else:
            raise ValueError(
                "Invalid representation. Choose 'CGR', 'connected_pair'"
            )

        self._add_dummy_nodes()

        edge_index = (
            torch.tensor(self.edge_index, dtype=torch.long).t().contiguous()
        )

        row, col = edge_index

        counter = 0
        for i, edge in enumerate(edge_index.t()):
            target_node = edge[0]

            incoming_edge_indices = (
                (col == target_node).nonzero(as_tuple=False).view(-1)
            )

            reverse_edge_mask = row[incoming_edge_indices] != edge[1]
            incoming_edge_indices = incoming_edge_indices[reverse_edge_mask]

            if len(incoming_edge_indices) > 0:
                incoming_edges_list.append(incoming_edge_indices)
                incoming_edges_batch.append(
                    torch.full_like(incoming_edge_indices, i)
                )
                incoming_edges_batch_from_zero.append(
                    torch.full_like(incoming_edge_indices, counter)
                )
                counter += 1

        self.incoming_edges_batch = torch.cat(incoming_edges_batch)
        self.incoming_edges_list = torch.cat(incoming_edges_list)
        self.incoming_edges_batch_from_zero = torch.cat(
            incoming_edges_batch_from_zero
        )

        self._compute_neighboring_nodes(edge_index)
        self._compute_incoming_edges_to_nodes()

    def _compute_incoming_edges_to_nodes(self):
        edge_index = (
            torch.tensor(self.edge_index, dtype=torch.long).t().contiguous()
        )
        row, col = edge_index

        incoming_edges_nodes_list = []
        incoming_edges_nodes_batch = []

        for node_idx in range(
            self.n_atoms * 2
        ):  # Iterate over all nodes (reactants and products)
            incoming_edge_indices = (
                (col == node_idx).nonzero(as_tuple=False).view(-1)
            )

            if len(incoming_edge_indices) > 0:
                incoming_edges_nodes_list.append(incoming_edge_indices)
                incoming_edges_nodes_batch.append(
                    torch.full_like(incoming_edge_indices, node_idx)
                )
            else:
                # If no incoming edges, append a placeholder (e.g., -1)
                incoming_edges_nodes_list.append(torch.tensor([-1]))
                incoming_edges_nodes_batch.append(torch.tensor([node_idx]))

        self.incoming_edges_nodes_list = torch.cat(incoming_edges_nodes_list)
        self.incoming_edges_nodes_batch = torch.cat(incoming_edges_nodes_batch)

    def _compute_neighboring_nodes(self, edge_index):
        row, col = edge_index

        # Initialize lists for neighboring nodes and their batch indicators
        neighboring_nodes_list = []
        neighboring_nodes_batch = []

        # Iterate over each node to collect neighbors
        for node_idx in range(
            self.n_atoms * 2
        ):  # Assuming we have reactants and products
            # Get the indices of the neighbors
            neighbor_indices = (
                (row == node_idx).nonzero(as_tuple=False).view(-1)
            )

            # Get the actual neighbors from col
            neighbors = col[neighbor_indices]

            if neighbors.numel() > 0:
                # Append neighbors and their batch indicators (current node index)
                neighboring_nodes_list.append(neighbors)
                neighboring_nodes_batch.append(
                    torch.full_like(neighbors, node_idx)
                )
            else:
                # If no neighbors, append the node itself
                neighboring_nodes_list.append(torch.tensor([node_idx]))
                neighboring_nodes_batch.append(torch.tensor([node_idx]))

        # Convert lists to tensors
        self.neighboring_nodes_list = (
            torch.cat(neighboring_nodes_list)
            if neighboring_nodes_list
            else torch.tensor([])
        )
        self.neighboring_nodes_batch = (
            torch.cat(neighboring_nodes_batch)
            if neighboring_nodes_batch
            else torch.tensor([])
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

    def _build_cgr(self):
        for i in range(self.n_atoms):
            self.f_atoms.append(self._get_atom_features(i))
            self.atom_origin_type.append(AtomOriginType.REACTANT_PRODUCT)

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

    def _add_dummy_nodes(self):
        if not self.dummy_node:
            return

        len_node_feat = len(
            self.atom_featurizer(self.mol_reac.GetAtomWithIdx(0))
        )
        len_bond_feat = len(self.bond_featurizer(None))

        if self.representation == "CGR":
            len_node_feat *= 2
            len_bond_feat *= 2

        if self.dummy_feat_init == "zeros":
            dummy_feature = torch.zeros(len_node_feat)
        else:
            dummy_feature = torch.ones(len_node_feat)

        f_bond = [0] * len_bond_feat

        if self.representation == "CGR" and self.dummy_node != "global":
            raise ValueError(
                "CGR representation only supports global dummy node"
            )

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

        dummy_connections = self.n_atoms
        if self.representation == "connected_pair":
            dummy_connections *= 2

        for i in range(dummy_connections):
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
        cache_graphs,
        max_cache_size,
        representation_cfg,
        transform_cfg,
    ):
        super(ChemDataset, self).__init__()
        self.smiles = smiles
        self.labels = labels
        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer
        self.cache_graphs = cache_graphs
        self.representation_cfg = representation_cfg
        self.transform_cfg = transform_cfg
        self.graph_cache = {}

        if cache_graphs:
            self.process_key = lru_cache(maxsize=max_cache_size)(
                self._process_key
            )
        else:
            self.process_key = self._process_key

    def _process_key(self, key):
        # TODO: add docstring
        smiles = self.smiles[key]
        molgraph = hydra.utils.instantiate(
            self.representation_cfg,
            smiles=smiles,
            atom_featurizer=self.atom_featurizer,
            bond_featurizer=self.bond_featurizer,
            transform_cfg=self.transform_cfg,
        )
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
        # data.is_real_bond = torch.tensor(
        #     molgraph.is_real_bond, dtype=torch.bool
        # )
        data.atom_origin_type = torch.tensor(
            molgraph.atom_origin_type, dtype=torch.long
        )
        # data.atom_origins = torch.tensor(
        #     molgraph.atom_origins, dtype=torch.long
        # )
        # data.incoming_edges_list = molgraph.incoming_edges_list
        # data.incoming_edges_batch = molgraph.incoming_edges_batch
        # data.incoming_edges_batch_from_zero = (
        #     molgraph.incoming_edges_batch_from_zero
        # )
        # data.neighboring_nodes_list = molgraph.neighboring_nodes_list
        # data.neighboring_nodes_batch = molgraph.neighboring_nodes_batch
        # data.incoming_edges_nodes_list = molgraph.incoming_edges_nodes_list
        # data.incoming_edges_nodes_batch = molgraph.incoming_edges_nodes_batch
        return data

    def get(self, key):
        # TODO: add docstring
        return self.process_key(key)

    def preprocess_all(self):
        for key in range(len(self.smiles)):
            self.process_key(key)

    def len(self):
        # TODO: add docstring
        return len(self.smiles)


def construct_loader(
    batch_size,
    num_workers,
    shuffle,
    split,
    cache_graphs,
    max_cache_size,
    preprocess_all,
    dataset_cfg,
    featurizer_cfg,
    representation_cfg,
    transform_cfg,
):
    
    dataset_cfg.use_fraction = dataset_cfg.get("use_fraction", False)  # Default to False if not in the YAML
    
    smiles, labels = load_csv_dataset(
        input_column=dataset_cfg.input_column,
        target_column=dataset_cfg.target_column,
        data_folder=dataset_cfg.data_folder,
        use_fraction=dataset_cfg.use_fraction,
        split=split,
    )

    atom_featurizer = make_featurizer(featurizer_cfg.atom_featurizer)
    bond_featurizer = make_featurizer(featurizer_cfg.bond_featurizer)

    dataset = ChemDataset(
        smiles=smiles,
        labels=labels,
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        cache_graphs=cache_graphs,
        max_cache_size=max_cache_size,
        representation_cfg=representation_cfg,
        transform_cfg=transform_cfg,
    )

    if preprocess_all:
        dataset.preprocess_all()

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
