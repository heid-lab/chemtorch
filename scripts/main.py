import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

import wandb
from deepreaction.utils import load_model, set_seed

OmegaConf.register_new_resolver("eval", eval)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # config mutable
    OmegaConf.set_struct(cfg, False)

    if getattr(cfg, "seed", None) is not None:
        set_seed(cfg.seed)

    if cfg.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    ############################# data instantiation #############################
    train_loader = hydra.utils.instantiate(cfg.data_cfg, shuffle=True, split="train")
    val_loader = hydra.utils.instantiate(cfg.data_cfg, shuffle=False, split="val")
    test_loader = hydra.utils.instantiate(cfg.data_cfg, shuffle=False, split="test")

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

    run_name = getattr(cfg, "run_name", None)

    # https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#utility-functions
    # check out
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)

    if cfg.wandb:
        wandb.init(
            project=cfg.project_name,
            group=cfg.group_name,
            name=run_name,
            config=resolved_cfg,
        )
        precompute_time = train_loader.dataset.precompute_time + val_loader.dataset.precompute_time + test_loader.dataset.precompute_time
        wandb.log(
            {"Precompute_time": precompute_time},
            commit=False,
        )

    print(OmegaConf.to_yaml(resolved_cfg))

    ############################# model instantiation #############################
    if cfg.use_loaded_model:
        model = hydra.utils.instantiate(cfg.model_cfg)

        if not os.path.exists(cfg.pretrained_path):
            raise ValueError(
                f"Pretrained model not found at {cfg.pretrained_path}"
            )

        model, _, _, _ = load_model(model, None, cfg.pretrained_path)
        model = model.to(device)

        try:
            sample_batch = next(iter(train_loader))
            model(sample_batch.to(device))
        except Exception as e:
            raise ValueError(
                f"Pretrained model incompatible with dataset: {str(e)}"
            )
    else:
        #### for models needing precomputed statistics on the dataset, e.g. PNA
        transform_cfg = getattr(cfg.data_cfg, "transform_cfg", None)
        if transform_cfg and hasattr(
            transform_cfg, "batched_degree_statistics"
        ):  # TODO: generalize
            model = hydra.utils.instantiate(
                cfg.model_cfg, dataset_precomputed=train_loader.dataset.statistics
            )
        else:
            model = hydra.utils.instantiate(cfg.model_cfg)
        model = model.to(device)

    total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Total parameters: {total_params:,}")

    under_parameters = getattr(cfg, "under_parameters", None)
    if under_parameters is not None and total_params > under_parameters:
        print(
            f"Model has {total_params:,} parameters, which exceeds the threshold of {under_parameters:,}. Skipping this run."
        )
        if cfg.wandb:
            wandb.log(
                {
                    "total_parameters": total_params,
                    "parameter_threshold_exceeded": True,
                }
            )
            wandb.run.summary["status"] = "parameter_threshold_exceeded"
        return False

    if cfg.wandb:
        wandb.log({"total_parameters": total_params}, commit=False)

    ############################# task instantiation #############################
    hydra.utils.instantiate(cfg.task_cfg, 
                            train_loader=train_loader, 
                            val_loader=val_loader, 
                            test_loader=test_loader, 
                            model=model,
                            device=device,
                            )
    #train()

    if cfg.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
