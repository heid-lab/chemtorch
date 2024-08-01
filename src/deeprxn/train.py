import torch
import numpy as np
from torch import nn
import math
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from tap import Tap
import random

from .data import Standardizer
from .model import GNN

#TODO: support cuda

class ArgumentParser(Tap):
    seed: int = 0  # Random seed for reproducibility
    epochs: int = 30  # Number of training epochs
    learning_rate: float = 0.001  # Learning rate for the optimizer
    data: str = "barriers_e2"  # Dataset to use (e.g., barriers_cycloadd, barriers_e2, barriers_rdb7, barriers_rgd1 ,barriers_sn2)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_epoch(model, loader, optimizer, loss, stdzer):
    #TODO: add docstring
    model.train()
    loss_all = 0

    for data in loader:
        optimizer.zero_grad()

        out = model(data)
        result = loss(out, stdzer(data.y))
        result.backward()

        optimizer.step()
        loss_all += loss(stdzer(out, rev=True), data.y)

    return math.sqrt(loss_all / len(loader.dataset))

def pred(model, loader, loss, stdzer):
    #TODO: add docstring
    model.eval()

    preds, ys = [], []
    with torch.no_grad():
        for data in loader:
            out = model(data)
            pred = stdzer(out, rev=True)
            preds.extend(pred.cpu().detach().tolist())

    return preds

def train(train_loader, val_loader, test_loader, args):
    #TODO add docstring
    #TODO add arguments for seed, epochs, learning rate, etc (currently hardcoded)
    #TODO add option for early stopping and implement accordingly (roll back to best model after some patience)
    mean = np.mean(train_loader.dataset.labels)
    std = np.std(train_loader.dataset.labels)
    stdzer = Standardizer(mean, std)

    model = GNN(train_loader.dataset.num_node_features, train_loader.dataset.num_edge_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = nn.MSELoss(reduction='sum')
    print(model)

    for epoch in range(0, 30):
        train_loss = train_epoch(model, train_loader, optimizer, loss, stdzer)
        preds = pred(model, val_loader, loss, stdzer)
        print("Epoch",epoch,"  Train RMSE", train_loss,"   Val RMSE", root_mean_squared_error(preds,val_loader.dataset.labels))

    preds = pred(model, test_loader, loss, stdzer)
    print("Test RMSE", root_mean_squared_error(preds,test_loader.dataset.labels))
    print("Test MAE", mean_absolute_error(preds,test_loader.dataset.labels))
