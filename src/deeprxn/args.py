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
