from rdkit import Chem
import torch
import torch_geometric as tg
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
import pandas as pd

from .featurizer import make_featurizer

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
    def __init__(self, smiles, atom_featurizer, bond_featurizer):
        self.smiles_reac, _, self.smiles_prod = smiles.split(">")
        self.f_atoms = []
        self.f_bonds = []
        self.edge_index = []

        mol_reac = make_mol(self.smiles_reac)
        mol_prod = make_mol(self.smiles_prod)

        ri2pi = map_reac_to_prod(mol_reac, mol_prod)
        n_atoms = mol_reac.GetNumAtoms()

        for i in range(n_atoms):
            f_atom_reac = atom_featurizer(mol_reac.GetAtomWithIdx(i))
            f_atom_prod = atom_featurizer(mol_prod.GetAtomWithIdx(ri2pi[i]))
            f_atom_diff = [y - x for x, y in zip(f_atom_reac, f_atom_prod)]
            f_atom = f_atom_reac + f_atom_diff
            self.f_atoms.append(f_atom)

            for j in range(i + 1, n_atoms):
                bond_reac = mol_reac.GetBondBetweenAtoms(i, j)
                bond_prod = mol_prod.GetBondBetweenAtoms(ri2pi[i], ri2pi[j])
                if bond_reac is None and bond_prod is None:
                    continue
                f_bond_reac = bond_featurizer(bond_reac)
                f_bond_prod = bond_featurizer(bond_prod)
                f_bond_diff = [y - x for x, y in zip(f_bond_reac, f_bond_prod)]
                f_bond = f_bond_reac + f_bond_diff
                self.f_bonds.append(f_bond)
                self.f_bonds.append(f_bond)
                self.edge_index.extend([(i, j), (j, i)])


class ChemDataset(Dataset):
    #TODO: add docstring
    #TODO: add functionality to drop invalid molecules
    #TODO: add option to cache graphs
    def __init__(self, smiles, labels, atom_featurizer, bond_featurizer, mode='mol'):
        super(ChemDataset, self).__init__()
        self.smiles = smiles
        self.labels = labels
        self.mode = mode
        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer

    def process_key(self, key):
        #TODO: add docstring
        smi = self.smiles[key]
        if self.mode == 'mol':
            molgraph = MolGraph(smi, self.atom_featurizer, self.bond_featurizer)
        elif self.mode == 'rxn':
            molgraph = RxnGraph(smi, self.atom_featurizer, self.bond_featurizer)
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

def load_from_csv(data_path, input_column='smiles', target_column='target'):
    #TODO: add docstring
    data_df = pd.read_csv(data_path)
    smiles = data_df[input_column].values
    labels = data_df[target_column].values.astype(float)
    return smiles, labels
    
def construct_loader(smiles, labels, atom_featurizer, bond_featurizer, shuffle=True, batch_size=50, mode='rxn'):
    #TODO: add docstring
    #TODO: add option for num_workers
    dataset = ChemDataset(smiles, labels, atom_featurizer, bond_featurizer, mode)
    loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=0,
                            pin_memory=True,
                            sampler=None)
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
