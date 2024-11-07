import operator

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from deeprxn.predict import predict_model
from deeprxn.train import train
from deeprxn.utils import set_seed

OmegaConf.register_new_resolver("eval", eval)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # config mutable
    OmegaConf.set_struct(cfg, False)
    set_seed(cfg.seed)

    if cfg.use_cuda and torch.cuda.is_available():
        cfg.device = "cuda"
    else:
        cfg.device = "cpu"

    print(f"Using device: {cfg.device}")

    print(OmegaConf.to_yaml(cfg))

    if cfg.wandb:
        wandb.init(
            project=cfg.project_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    train_loader = hydra.utils.instantiate(
        cfg.data, shuffle=True, split="train"
    )
    val_loader = hydra.utils.instantiate(cfg.data, shuffle=False, split="val")
    test_loader = hydra.utils.instantiate(
        cfg.data, shuffle=False, split="test"
    )

    if cfg.mode == "train":
        train(train_loader, val_loader, test_loader, cfg)
    elif cfg.mode == "predict":
        predict_model(test_loader, cfg)
    else:
        raise ValueError(
            f"Invalid mode: {cfg.mode}. Choose 'train' or 'predict'."
        )

    if cfg.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
