# test.py
from deeprxn.data import load_from_csv, construct_loader, Standardizer
from deeprxn.train import train, load_model, predict
from deeprxn.featurizer import make_featurizer
from deeprxn.utils import set_seed
from deeprxn.model import GNN
from deeprxn.args import ArgumentParser
import numpy as np

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

    if args.mode == "train":
        train(train_loader, val_loader, test_loader, args)

    elif args.mode == "predict":
        model = GNN(train_loader.dataset.num_node_features, train_loader.dataset.num_edge_features)
        model, _, _, _ = load_model(model, None, args.model_path)
        
        # Make predictions on the test set
        stdzer = Standardizer(np.mean(train_loader.dataset.labels), np.std(train_loader.dataset.labels))
        test_preds = predict(model, test_loader, stdzer)
        
        # Print or save predictions as needed
        print("Test set predictions:", test_preds)
    else:
        raise ValueError(f"Invalid mode: {args.mode}. Choose 'train' or 'predict'.")

if __name__ == "__main__":
    main()
