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

    # print(OmegaConf.to_yaml(cfg))

    train_loader = hydra.utils.instantiate(
        cfg.data, shuffle=True, split="train"
    )
    val_loader = hydra.utils.instantiate(cfg.data, shuffle=False, split="val")
    test_loader = hydra.utils.instantiate(
        cfg.data, shuffle=False, split="test"
    )

    OmegaConf.update(
        cfg,
        "num_node_features",
        train_loader.dataset.num_node_features,
        merge=True,
    )
    OmegaConf.update(
        cfg,
        "num_edge_features",
        train_loader.dataset.num_edge_features,
        merge=True,
    )

    OmegaConf.update(
        cfg,
        "model_path",
        f"{cfg.model_path}_{cfg.project_name}_{cfg.data.dataset_cfg.data_folder}_{cfg.seed}",
        merge=True,
    )

    # https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#utility-functions
    # check out
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)

    if cfg.wandb:
        wandb.init(
            project=cfg.project_name,
            config=resolved_cfg,
        )
        wandb.log(
            {"train_precompute_time": train_loader.dataset.precompute_time}
        )
        # wandb.log({"val_precompute_time": val_loader.dataset.precompute_time})
        # wandb.log(
        #     {"test_precompute_time": test_loader.dataset.precompute_time}
        # )

    print(OmegaConf.to_yaml(resolved_cfg))

    if cfg.mode == "train":
        train(
            train_loader,
            val_loader,
            test_loader,
            pretrained_path=cfg.pretrained_path,
            cfg=cfg,
            finetune=False,
        )
    elif cfg.mode == "finetune":
        if not cfg.pretrained_path:
            raise ValueError(
                "pretrained_path must be specified for finetuning"
            )
        train(
            train_loader,
            val_loader,
            test_loader,
            pretrained_path=cfg.pretrained_path,
            cfg=cfg,
            finetune=True,
        )
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
