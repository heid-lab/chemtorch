from deeprxn.data import load_from_csv, construct_loader
from deeprxn.train import train
from deeprxn.featurizer import make_featurizer

atom_featurizer = make_featurizer("atom_rdkit_organic")
bond_featurizer = make_featurizer("bond_rdkit_base")

'''
#Example for molecules:
smiles, labels = load_from_csv("https://github.com/hesther/rxn_workshop/raw/main/data/esol/train_full.csv", 'smiles', 'logSolubility')
train_loader = construct_loader(smiles, labels, atom_featurizer, bond_featurizer, True, mode='mol')


smiles, labels = load_from_csv("https://github.com/hesther/rxn_workshop/raw/main/data/esol/val_full.csv", 'smiles', 'logSolubility')
val_loader = construct_loader(smiles, labels, atom_featurizer, bond_featurizer, False, mode='mol')

smiles, labels = load_from_csv("https://github.com/hesther/rxn_workshop/raw/main/data/esol/test_full.csv", 'smiles', 'logSolubility')
test_loader = construct_loader(smiles, labels, atom_featurizer, bond_featurizer, False, mode='mol')

train(train_loader, val_loader, test_loader)
'''


#Example for reactions:
smiles, labels = load_from_csv("https://github.com/hesther/rxn_workshop/raw/main/data/e2/train_full.csv", 'AAM', 'ea')
train_loader = construct_loader(smiles, labels, atom_featurizer, bond_featurizer, True, mode='rxn')


smiles, labels = load_from_csv("https://github.com/hesther/rxn_workshop/raw/main/data/e2/val_full.csv", 'AAM', 'ea')
val_loader = construct_loader(smiles, labels, atom_featurizer, bond_featurizer, False, mode='rxn')

smiles, labels = load_from_csv("https://github.com/hesther/rxn_workshop/raw/main/data/e2/test_full.csv", 'AAM', 'ea')
test_loader = construct_loader(smiles, labels, atom_featurizer, bond_featurizer, False, mode='rxn')

train(train_loader, val_loader, test_loader)


#TODO: Test also on bigger datasets eg. ccsd barrier heights, compare to chemprop v1 and v2
