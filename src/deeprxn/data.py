import time
from functools import lru_cache
from typing import Literal, Optional

import hydra
import torch
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
        enthalpy=None,
    ):
        super(ChemDataset, self).__init__()
        self.smiles = smiles
        self.labels = labels
        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer
        self.cache_graphs = cache_graphs
        self.representation_cfg = representation_cfg
        self.transform_cfg = transform_cfg
        self.enthalpy = enthalpy
        self.precompute_time = 0
        self.graph_cache = {}

        if cache_graphs:
            self.process_key = lru_cache(maxsize=max_cache_size)(
                self._process_key
            )
        else:
            self.process_key = self._process_key

        self.graph_transforms = []
        self.dataset_transforms = []
        if self.transform_cfg is not None:
            for _, config in self.transform_cfg.items():
                transform = hydra.utils.instantiate(config)
                if (
                    config.type == "graph"
                ):  # TODO: look into making all transforms being performed on dataset
                    self.graph_transforms.append(transform)
                elif config.type == "dataset":
                    self.dataset_transforms.append(transform)
                else:
                    assert False, f"Unknown transform type: {config.type}"

    def _process_key(self, key):
        # TODO: add docstring
        smiles = self.smiles[key]
        label = self.labels[key]
        enthalpy_value = (
            self.enthalpy[key] if self.enthalpy is not None else None
        )
        molgraph = hydra.utils.instantiate(
            self.representation_cfg,
            smiles=smiles,
            label=label,
            enthalpy=enthalpy_value,
            atom_featurizer=self.atom_featurizer,
            bond_featurizer=self.bond_featurizer,
        )
        molgraph_tg_data_obj = (
            molgraph.to_pyg_data()
        )  # TODO: look into making representations inherit from PyG Data

        if self.transform_cfg is not None:
            for transform in self.graph_transforms:
                molgraph_tg_data_obj = transform(molgraph_tg_data_obj)

        return molgraph_tg_data_obj

    def get(self, key):
        # TODO: add docstring
        return self.process_key(key)

    def preprocess_all(self):
        start_time = time.time()
        for key in range(len(self.smiles)):
            self.process_key(key)
        self.precompute_time = time.time() - start_time

    def len(self):
        # TODO: add docstring
        return len(self.smiles)


def construct_loader(
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    split: Literal["train", "val", "test"],
    cache_graphs: bool,
    max_cache_size: int,
    preprocess_all: bool,
    dataset_cfg: dict,
    featurizer_cfg: dict,
    representation_cfg: dict,
    transform_cfg: Optional[dict] = None,
) -> DataLoader:
    """Construct a PyTorch Geometric DataLoader with specified configuration."""

    split_params = {
        "train_ratio": dataset_cfg.get("train_ratio", 0.8),
        "val_ratio": dataset_cfg.get("val_ratio", 0.1),
        "test_ratio": dataset_cfg.get("test_ratio", 0.1),
        "use_pickle": dataset_cfg.get("use_pickle", False),
        "use_enthalpy": dataset_cfg.get("use_enthalpy", False),
        "enthalpy_column": dataset_cfg.get("enthalpy_column", None),
    }

    data = load_csv_dataset(
        input_column=dataset_cfg.input_column,
        target_column=dataset_cfg.target_column,
        data_folder=dataset_cfg.data_folder,
        reduced_dataset=dataset_cfg.reduced_dataset,
        split=split,
        **split_params,
    )

    smiles = data[0]
    labels = data[1]
    enthalpy = data[2] if len(data) > 2 else None

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
        enthalpy=enthalpy,
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

    # this code is needed for PNA
    dataset_statistics = {}
    if dataset.dataset_transforms:
        original_state = loader.generator.get_state()
        for batch in loader:
            for transform in dataset.dataset_transforms:
                batch = transform(batch)

        for transform in dataset.dataset_transforms:  # TODO: make this nicer
            if transform.needs_second_dataloader:
                stats = transform.finalize(loader)
            else:
                stats = transform.finalize()
            dataset_statistics.update(stats)

        loader.generator.set_state(original_state)

    dataset.statistics = dataset_statistics

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
