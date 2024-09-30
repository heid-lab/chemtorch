import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from deeprxn.train import train
from deeprxn.utils import set_seed


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

    if cfg.wandb:
        wandb.init(
            project=cfg.project_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    train_loader = hydra.utils.instantiate(
        cfg.transformation, shuffle=True, split="train"
    )
    val_loader = hydra.utils.instantiate(
        cfg.transformation, shuffle=False, split="val"
    )
    test_loader = hydra.utils.instantiate(
        cfg.transformation, shuffle=False, split="test"
    )

    if cfg.mode == "train":
        train(train_loader, val_loader, test_loader, cfg)

    # TODO: Implement predict mode

    else:
        raise ValueError(
            f"Invalid mode: {cfg.mode}. Choose 'train' or 'predict'."
        )

    if cfg.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
