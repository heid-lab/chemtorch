from tap import Tap

class ArgumentParser(Tap):
    seed: int = 0  # Random seed for reproducibility
    epochs: int = 30  # Number of training epochs
    learning_rate: float = 0.001  # Learning rate for the optimizer
    data: str = "barriers_e2"  # Dataset to use (e.g., barriers_cycloadd, barriers_e2, barriers_rdb7, barriers_rgd1, barriers_sn2)    
    model_path: str = "model.pt"  # Path to save/load the model
    mode: str = "train"  # Mode: 'train' or 'predict'
    patience: int = 10  # Number of epochs to wait for improvement before early stopping
    min_delta: float = 0.001  # Minimum change in validation loss to qualify as an improvement
    use_cuda: bool = False  # Enable CUDA if available
    num_workers: int = 0  # Number of workers for data loader
    atom_featurizer: str = "atom_rdkit_organic"  # Atom featurizer option
    bond_featurizer: str = "bond_rdkit_base"  # Bond featurizer option
    representation: str = "CGR"  # Molecular representation option
    save_model: bool = False  # Save the model after training
    connection_direction: str = "bidirectional"  # Connection direction for the CGR representation: reactants_to_products, products_to_reactants, bidirectional
    pool_type: str = "global"  # global, reactants, products
