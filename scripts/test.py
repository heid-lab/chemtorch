from deeprxn.data import load_from_csv, construct_loader, Standardizer
from deeprxn.train import train, load_model, predict
from deeprxn.featurizer import make_featurizer
from deeprxn.utils import set_seed
from deeprxn.model import GNN
from deeprxn.args import ArgumentParser
import numpy as np
import torch

def main():
    args = ArgumentParser().parse_args()
    
    set_seed(args.seed)

    args.device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {args.device}")

    valid_atom_featurizers = ["atom_atomic_num", "atom_rdkit_base", "atom_rdkit_organic"]
    valid_bond_featurizers = ["bond_rdkit_base"]
    
    if args.atom_featurizer not in valid_atom_featurizers:
        raise ValueError(f"Invalid atom featurizer: {args.atom_featurizer}. Choose from {valid_atom_featurizers}")
    
    if args.bond_featurizer not in valid_bond_featurizers:
        raise ValueError(f"Invalid bond featurizer: {args.bond_featurizer}. Choose from {valid_bond_featurizers}")

    atom_featurizer = make_featurizer(args.atom_featurizer)
    bond_featurizer = make_featurizer(args.bond_featurizer)

    # Load data and construct loaders
    smiles, labels = load_from_csv(args.data, "train")
    train_loader = construct_loader(smiles, labels, atom_featurizer, bond_featurizer, args.num_workers, True, mode='rxn', representation=args.representation)

    smiles, labels = load_from_csv(args.data, "val")
    val_loader = construct_loader(smiles, labels, atom_featurizer, bond_featurizer, args.num_workers, False, mode='rxn', representation=args.representation)

    smiles, labels = load_from_csv(args.data, "test")
    test_loader = construct_loader(smiles, labels, atom_featurizer, bond_featurizer, args.num_workers, False, mode='rxn', representation=args.representation)

    if args.mode == "train":
        train(train_loader, val_loader, test_loader, args)

    elif args.mode == "predict":
        model = GNN(train_loader.dataset.num_node_features, train_loader.dataset.num_edge_features)
        model, _, _, _ = load_model(model, None, args.model_path)
        model = model.to(args.device)
        
        # Make predictions on the test set
        stdzer = Standardizer(np.mean(train_loader.dataset.labels), np.std(train_loader.dataset.labels))
        test_preds = predict(model, test_loader, stdzer, args.device)
        
        # Print or save predictions as needed
        print("Test set predictions:", test_preds)
    else:
        raise ValueError(f"Invalid mode: {args.mode}. Choose 'train' or 'predict'.")

if __name__ == "__main__":
    main()
