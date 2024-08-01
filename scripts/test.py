import os
from deeprxn.data import load_from_csv, construct_loader
from deeprxn.train import train, ArgumentParser, set_seed
from deeprxn.featurizer import make_featurizer

def get_data_paths(data_folder):
    base_path = os.path.join("data", data_folder)
    return {
        "train": os.path.join(base_path, "train.csv"),
        "val": os.path.join(base_path, "val.csv"),
        "test": os.path.join(base_path, "test.csv")
    }

def main():
    args = ArgumentParser().parse_args()
    
    set_seed(args.seed)

    atom_featurizer = make_featurizer("atom_rdkit_organic")
    bond_featurizer = make_featurizer("bond_rdkit_base")

    data_paths = get_data_paths(args.data)

    # Load data and construct loaders
    smiles, labels = load_from_csv(data_paths["train"], 'AAM', 'ea')
    train_loader = construct_loader(smiles, labels, atom_featurizer, bond_featurizer, True, mode='rxn')

    smiles, labels = load_from_csv(data_paths["val"], 'AAM', 'ea')
    val_loader = construct_loader(smiles, labels, atom_featurizer, bond_featurizer, False, mode='rxn')

    smiles, labels = load_from_csv(data_paths["test"], 'AAM', 'ea')
    test_loader = construct_loader(smiles, labels, atom_featurizer, bond_featurizer, False, mode='rxn')

    train(train_loader, val_loader, test_loader, args)

if __name__ == "__main__":
    main()
