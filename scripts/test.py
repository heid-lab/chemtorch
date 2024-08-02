from deeprxn.data import load_from_csv, construct_loader
from deeprxn.train import train, ArgumentParser, set_seed
from deeprxn.featurizer import make_featurizer

def main():
    args = ArgumentParser().parse_args()
    
    set_seed(args.seed)

    atom_featurizer = make_featurizer("atom_rdkit_organic")
    bond_featurizer = make_featurizer("bond_rdkit_base")

    # Load data and construct loaders
    smiles, labels = load_from_csv(args.data, "train")
    train_loader = construct_loader(smiles, labels, atom_featurizer, bond_featurizer, True, mode='rxn')

    smiles, labels = load_from_csv(args.data, "val")
    val_loader = construct_loader(smiles, labels, atom_featurizer, bond_featurizer, False, mode='rxn')

    smiles, labels = load_from_csv(args.data, "test")
    test_loader = construct_loader(smiles, labels, atom_featurizer, bond_featurizer, False, mode='rxn')

    train(train_loader, val_loader, test_loader, args)

if __name__ == "__main__":
    main()
