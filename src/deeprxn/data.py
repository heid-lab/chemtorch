from rdkit import Chem
import torch
import torch_geometric as tg
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
import pandas as pd
import os
from deeprxn.featurizer import make_featurizer

def make_mol(smi):
    #TODO: add docstring
    params = Chem.SmilesParserParams()
    params.removeHs = False
    return Chem.MolFromSmiles(smi,params)

def map_reac_to_prod(mol_reac, mol_prod):
    #TODO: add docstring
    prod_map_to_id = dict([(atom.GetAtomMapNum(),atom.GetIdx()) for atom in mol_prod.GetAtoms()])
    reac_id_to_prod_id = dict([(atom.GetIdx(),prod_map_to_id[atom.GetAtomMapNum()]) for atom in mol_reac.GetAtoms()])
    return reac_id_to_prod_id

class MolGraph:
    #TODO: add docstring
    def __init__(self, smiles, atom_featurizer, bond_featurizer):
        self.smiles = smiles
        self.f_atoms = []
        self.f_bonds = []
        self.edge_index = []

        mol = make_mol(self.smiles)
        n_atoms=mol.GetNumAtoms()

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
    #TODO: add docstring
    #TODO (maybe): add support for unbalanced reactions?
    def __init__(self, smiles, atom_featurizer, bond_featurizer, representation="CGR"):
        self.smiles_reac, _, self.smiles_prod = smiles.split(">")
        self.f_atoms = []
        self.f_bonds = []
        self.edge_index = []

        self.mol_reac = make_mol(self.smiles_reac)
        self.mol_prod = make_mol(self.smiles_prod)
        self.ri2pi = map_reac_to_prod(self.mol_reac, self.mol_prod)
        self.n_atoms = self.mol_reac.GetNumAtoms()

        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer
        self.representation = representation

        if self.representation == "CGR":
            self._build_cgr()
        elif self.representation == "fully_connected":
            self._build_fully_connected()
        elif representation == "connected_pair":
            self._build_connected_pair()
        else:
            raise ValueError("Invalid representation. Choose 'CGR', 'fully_connected', or 'connected_pair'.")

    def _get_atom_features(self, i):
        f_atom_reac = self.atom_featurizer(self.mol_reac.GetAtomWithIdx(i))
        f_atom_prod = self.atom_featurizer(self.mol_prod.GetAtomWithIdx(self.ri2pi[i]))
        f_atom_diff = [y - x for x, y in zip(f_atom_reac, f_atom_prod)]
        return f_atom_reac + f_atom_diff

    def _get_bond_features(self, bond_reac, bond_prod):
        f_bond_reac = self.bond_featurizer(bond_reac) if bond_reac else [0] * len(self.bond_featurizer(None))
        f_bond_prod = self.bond_featurizer(bond_prod) if bond_prod else [0] * len(self.bond_featurizer(None))
        f_bond_diff = [y - x for x, y in zip(f_bond_reac, f_bond_prod)]
        return f_bond_reac + f_bond_diff

    def _build_cgr(self):
        for i in range(self.n_atoms):
            self.f_atoms.append(self._get_atom_features(i))

            for j in range(i + 1, self.n_atoms):
                bond_reac = self.mol_reac.GetBondBetweenAtoms(i, j)
                bond_prod = self.mol_prod.GetBondBetweenAtoms(self.ri2pi[i], self.ri2pi[j])
                if bond_reac is None and bond_prod is None:
                    continue
                f_bond = self._get_bond_features(bond_reac, bond_prod)
                self.f_bonds.append(f_bond)
                self.f_bonds.append(f_bond)
                self.edge_index.extend([(i, j), (j, i)])

    def _build_fully_connected(self):
        for i in range(self.n_atoms):
            self.f_atoms.append(self._get_atom_features(i))

            for j in range(self.n_atoms):
                if i != j:
                    bond_reac = self.mol_reac.GetBondBetweenAtoms(i, j)
                    bond_prod = self.mol_prod.GetBondBetweenAtoms(self.ri2pi[i], self.ri2pi[j])
                    f_bond = self._get_bond_features(bond_reac, bond_prod)
                    self.f_bonds.append(f_bond)
                    self.edge_index.append((i, j))

    def _build_connected_pair(self):
        # Build reactant graph
        for i in range(self.n_atoms):
            self.f_atoms.append(self.atom_featurizer(self.mol_reac.GetAtomWithIdx(i)))
            for j in range(i + 1, self.n_atoms):
                bond_reac = self.mol_reac.GetBondBetweenAtoms(i, j)
                if bond_reac:
                    f_bond = self.bond_featurizer(bond_reac)
                    self.f_bonds.append(f_bond)
                    self.f_bonds.append(f_bond)
                    self.edge_index.extend([(i, j), (j, i)])

        # Build product graph
        offset = self.n_atoms
        for i in range(self.n_atoms):
            self.f_atoms.append(self.atom_featurizer(self.mol_prod.GetAtomWithIdx(self.ri2pi[i])))
            for j in range(i + 1, self.n_atoms):
                bond_prod = self.mol_prod.GetBondBetweenAtoms(self.ri2pi[i], self.ri2pi[j])
                if bond_prod:
                    f_bond = self.bond_featurizer(bond_prod)
                    self.f_bonds.append(f_bond)
                    self.f_bonds.append(f_bond)
                    self.edge_index.extend([(i + offset, j + offset), (j + offset, i + offset)])

        # Connect corresponding atoms between reactants and products
        for i in range(self.n_atoms):
            f_bond = [0] * len(self.bond_featurizer(None))  # Use a zero vector for the connecting edge
            self.f_bonds.append(f_bond)
            self.f_bonds.append(f_bond)
            self.edge_index.extend([(i, i + offset), (i + offset, i)])


class ChemDataset(Dataset):
    #TODO: add docstring
    #TODO: add functionality to drop invalid molecules
    #TODO: add option to cache graphs
    def __init__(self, smiles, labels, atom_featurizer, bond_featurizer, mode='mol', representation="CGR"):
        super(ChemDataset, self).__init__()
        self.smiles = smiles
        self.labels = labels
        self.mode = mode
        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer
        self.representation = representation 

    def process_key(self, key):
        #TODO: add docstring
        smi = self.smiles[key]
        if self.mode == 'mol':
            molgraph = MolGraph(smi, self.atom_featurizer, self.bond_featurizer)
        elif self.mode == 'rxn':
            molgraph = RxnGraph(smi, self.atom_featurizer, self.bond_featurizer, self.representation)
        else:
            raise ValueError("Unknown option for mode", self.mode)
        mol = self.molgraph2data(molgraph, key)
        return mol

    def molgraph2data(self, molgraph, key):
        #TODO: add docstring
        data = tg.data.Data()
        data.x = torch.tensor(molgraph.f_atoms, dtype=torch.float)
        data.edge_index = torch.tensor(molgraph.edge_index, dtype=torch.long).t().contiguous()
        data.edge_attr = torch.tensor(molgraph.f_bonds, dtype=torch.float)
        data.y = torch.tensor([self.labels[key]], dtype=torch.float)
        data.smiles = self.smiles[key]

        return data

    def get(self,key):
        #TODO: add docstring
        return self.process_key(key)

    def len(self):
        #TODO: add docstring
        return len(self.smiles)

def get_data_paths(data_folder):
    base_path = os.path.join("data", data_folder)
    return {
        "train": os.path.join(base_path, "train.csv"),
        "val": os.path.join(base_path, "val.csv"),
        "test": os.path.join(base_path, "test.csv")
    }

def load_from_csv(dataset_name, split):
    #TODO: add docstring

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
    
def construct_loader(smiles, 
                     labels, 
                     atom_featurizer, 
                     bond_featurizer, 
                     num_workers,
                     shuffle=True, 
                     batch_size=50, 
                     mode='rxn', 
                     representation="CGR"):
    #TODO: add docstring
    dataset = ChemDataset(smiles, labels, atom_featurizer, bond_featurizer, mode, representation)
    loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            pin_memory=True,
                            sampler=None,
                            generator=torch.Generator().manual_seed(0))
    return loader

class Standardizer:
    #TODO: add docstring
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x, rev=False):
        if rev:
            return (x * self.std) + self.mean
        return (x - self.mean) / self.std
