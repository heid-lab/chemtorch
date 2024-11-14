from enum import IntEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal, Tuple

import hydra
import torch
import torch_geometric as tg
from rdkit import Chem
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from deeprxn.featurizer.featurizer import make_featurizer
from deeprxn.utils import load_csv_dataset


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
    transform_cfg=None,
):

    smiles, labels = load_csv_dataset(
        input_column=dataset_cfg.input_column,
        target_column=dataset_cfg.target_column,
        data_folder=dataset_cfg.data_folder,
        reduced_dataset=dataset_cfg.reduced_dataset,
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
