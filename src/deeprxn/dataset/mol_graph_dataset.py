import time
import hydra
import pandas as pd
import torch
import torch_geometric as tg

from torch_geometric.data import Data, DataLoader, Dataset
from functools import lru_cache
from typing import Optional

# TODO: Remove dependency on hydra
# TODO: PASS ONLY PARAMETERS NEEDED FOR THE SPECIFIC REPRESENTATION
# Problem: Representations are coupled to featurizers
# Possible solution: pass featurizers as dict and raise errors when an expected featurizer is not present,
class MolGraphDataset(tg.data.Dataset):
    def __init__(
            self,
            data: pd.DataFrame,
            input_column: str,
            target_column: str,
            # TODO: move featurizers to representation
            atom_featurizer: Optional[callable],    
            bond_featurizer: Optional[callable],    
            qm_featurizer: Optional[callable],      
            single_featurizer: Optional[callable],  
            representation_cfg: Optional[dict],     # dependency on hydra
            transform_cfg: Optional[dict],          # dependency on hydra
            subsample: Optional[int|float] = None,
            # TODO: Remove redundant args
            cache_graphs: Optional[bool] = True,
            max_cache_size: Optional[int] = None,
            preprocess_all: Optional[bool] = True,
            *args,      # ignore additional positional arguments
            **kwargs    # ignore additional keyword arguments
    ):
        """
        Initializes the MolGraphDataset with the provided data.

        Args:
            data (pd.DataFrame): The input data containing molecular graphs.
        """
        super().__init__()
        if subsample is not None:
            if isinstance(subsample, int):
                data = data.sample(n=subsample)
            elif isinstance(subsample, float):
                data = data.sample(frac=subsample)
            else:
                raise ValueError("subsample must be an int or a float")
        
        self.smiles_strs = data[input_column]
        self.labels = data[target_column]

        self.cache_graphs = cache_graphs
        self.graph_cache = {}
        
        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer
        self.qm_featurizer = qm_featurizer
        self.single_featurizer = single_featurizer

        self.representation_cfg = representation_cfg
        self.transform_cfg = transform_cfg
        self.precompute_time = 0

        # SETUP OPTIONAL CACHING FOR PROCESSING FUNCTION 
        if cache_graphs:
            self.process_sample = lru_cache(maxsize=max_cache_size)(
                self._process_sample
            )
        else:
            self.process_sample = self._process_sample

        # CREATE TRANSOFRM LISTS DEFINED IN THE CONFIG
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
        
        if cache_graphs and preprocess_all:
            # Only works if caching is enabled
            self.preprocess_all()


    def _process_sample(self, idx) -> Data:
        """
        Process a single sample from the dataset and convert it into a PyTorch Geometric `Data` object.

        This method performs the following steps:
        1. Retrieves the SMILES string, label, and optional enthalpy value for the molecule at the specified index.
        2. Constructs a molecular graph representation using the configuration provided in `:attr:self.representation_cfg`.
        3. Converts the molecular graph into a PyTorch Geometric `Data` object.
        4. Applies any graph-level transformations specified in `:attr:self.transform_cfg`.

        Args:
            idx (int): The index of the sample to process.

        Returns:
            tg.data.Data: A PyTorch Geometric `Data` object representing the processed molecular graph.

        """
        smiles_str = self.smiles_strs[idx]
        label = self.labels[idx]

        try:       
            repr = hydra.utils.instantiate(
                self.representation_cfg,
                smiles=smiles_str,
                label=label,
                atom_featurizer=self.atom_featurizer,
                bond_featurizer=self.bond_featurizer,
                qm_featurizer=self.qm_featurizer,
                single_featurizer=self.single_featurizer,
            )
        except Exception as e:
            raise ValueError(
                f"Error processing sample {idx}, Error: {str(e)}"
            )

        # TODO: look into making representations inherit from PyG Data
        graph = repr.to_pyg_data()

        if self.transform_cfg is not None:
            for transform in self.graph_transforms:
                graph = transform(graph)

        return graph


    def get(self, idx) -> Data:
        """
        Retrieve a processed data object by its index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tg.data.Data: A PyTorch Geometric `Data` object representing the processed molecular graph.
        """
        return self.process_sample(idx)


    def preprocess_all(self) -> None:
        """
        Preprocess all samples in the dataset and store them in the cache.

        Raises:
            RuntimeError: If caching is not enabled (see `:func:self.__init__()` for how to enable caching).
        """
        if not self.cache_graphs:
            raise RuntimeError("preprocess_all can only be called if caching is enabled.")

        start_time = time.time()
        for idx in range(len(self.smiles_strs)):
            self.process_sample(idx)
        self.precompute_time = time.time() - start_time


    def len(self):
        """Return the number of samples in the dataset."""
        return len(self.smiles_strs)


# TODO: Istantiate dataloader via hydra
def construct_loader(
        dataset: Dataset,
        batch_size: int,   
        shuffle: bool,
        num_workers: int,
        pin_memory: bool = True,
        sampler: Optional[torch.utils.data.Sampler] = None,
        generator: Optional[torch.Generator] = torch.Generator(),
    ):
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        sampler=None,
        generator=generator.manual_seed(0), # TODO: DO NOT HARDCODE!!!
    )

    # TODO: DO NOT HARD CODE THIS, this should be part of the PNA model
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



