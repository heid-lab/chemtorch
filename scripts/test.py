import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from deeprxn.data import load_from_csv, construct_loader, Standardizer
from deeprxn.train import train, load_model, predict
from deeprxn.featurizer import make_featurizer
from deeprxn.utils import set_seed
from deeprxn.model import GNN
import numpy as np
import torch

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # config mutable
    OmegaConf.set_struct(cfg, False)

    set_seed(cfg.seed)

    if cfg.use_cuda and torch.cuda.is_available():
        cfg.device = 'cuda'
    else:
        cfg.device = 'cpu'

    print(f"Using device: {cfg.device}")

    atom_featurizer = make_featurizer(cfg.features.atom_featurizer)
    bond_featurizer = make_featurizer(cfg.features.bond_featurizer)

    # Load data and construct loaders
    smiles, labels = load_from_csv(cfg.data.name, "train")
    train_loader = construct_loader(smiles, 
                                    labels, 
                                    atom_featurizer, 
                                    bond_featurizer, 
                                    cfg.num_workers, 
                                    True, 
                                    mode='rxn', 
                                    representation=cfg.transformation.representation, 
                                    connection_direction=cfg.transformation.connection_direction,
                                    dummy_node=cfg.transformation.dummy_node,
                                    dummy_connection_direction=cfg.transformation.dummy_connection_direction)

    smiles, labels = load_from_csv(cfg.data.name, "val")
    val_loader = construct_loader(smiles, 
                                  labels, 
                                  atom_featurizer, 
                                  bond_featurizer, 
                                  cfg.num_workers, 
                                  False, 
                                  mode='rxn', 
                                  representation=cfg.transformation.representation, 
                                  connection_direction=cfg.transformation.connection_direction,
                                  dummy_node=cfg.transformation.dummy_node,
                                  dummy_connection_direction=cfg.transformation.dummy_connection_direction)

    smiles, labels = load_from_csv(cfg.data.name, "test")
    test_loader = construct_loader(smiles, 
                                   labels, 
                                   atom_featurizer, 
                                   bond_featurizer, 
                                   cfg.num_workers, 
                                   False, 
                                   mode='rxn', 
                                   representation=cfg.transformation.representation, 
                                   connection_direction=cfg.transformation.connection_direction,
                                   dummy_node=cfg.transformation.dummy_node,
                                   dummy_connection_direction=cfg.transformation.dummy_connection_direction)
    
    if cfg.mode == "train":
        cfg.model.num_node_features = train_loader.dataset.num_node_features
        cfg.model.num_edge_features = train_loader.dataset.num_edge_features

        train(train_loader, val_loader, test_loader, cfg)

    elif cfg.mode == "predict":
        cfg.model.num_node_features = train_loader.dataset.num_node_features
        cfg.model.num_edge_features = train_loader.dataset.num_edge_features

        model = GNN(train_loader.dataset.num_node_features, train_loader.dataset.num_edge_features)
        model, _, _, _ = load_model(model, None, cfg.model_path)
        model = model.to(cfg.device)
        
        # Make predictions on the test set
        stdzer = Standardizer(np.mean(train_loader.dataset.labels), np.std(train_loader.dataset.labels))
        test_preds = predict(model, test_loader, stdzer, cfg.device)
        
        # Print or save predictions as needed
        print("Test set predictions:", test_preds)
    else:
        raise ValueError(f"Invalid mode: {cfg.mode}. Choose 'train' or 'predict'.")

if __name__ == "__main__":
    main()
