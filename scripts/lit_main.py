import os

import hydra
from hydra.utils import instantiate
from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf

import wandb
from deepreaction.data_module_lightning import DataModule

OmegaConf.register_new_resolver("eval", eval)   # TODO: What is this?


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # config mutable
    OmegaConf.set_struct(cfg, False)

    seed = getattr(cfg, "seed", 0)
    seed_everything(seed)

    ##### DATA MODULE ##############################################################
    data_pipeline = instantiate(cfg.data_pipeline)
    dataset_factory = instantiate(cfg.dataset)
    dataloader_factory = instantiate(cfg.dataloader)
    data_module = DataModule(
        data_pipeline=data_pipeline,
        dataset_factory=dataset_factory,
        dataloader_factory=dataloader_factory,
    )

    ##### UPDATE GLOBAL CONFIG FROM DATASET ATTRIBUTES ##############################
    dataset_properties = cfg.get("runtime_agrs_from_train_dataset_props", [])
    if dataset_properties:
        print(
            "INFO: Updating global config with properties of training dataset"
        )
        for dataset_property in dataset_properties:
            OmegaConf.update(
                cfg=cfg, 
                key=dataset_property, 
                value=data_module.get_dataset_property(key='train', property=dataset_property),
                merge=True
            )

    run_name = getattr(cfg, "run_name", None)
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)

    # print(f"INFO: Final config:\n{OmegaConf.to_yaml(resolved_cfg)}")

    ##### INITIALIZE W&B ##########################################################
    if cfg.log:
        wandb.init(
            project=cfg.project_name,
            group=cfg.group_name,
            name=run_name,
            config=resolved_cfg,
        )
        for stage in ['train', 'val', 'test']:
            precompute_time = data_module.get_dataset_property(stage, 'precompute_time')
            wandb.log({f"{stage}_precompute_time": precompute_time}, commit=False,)


    ##### MODEL ##################################################################
    model = instantiate(cfg.model)
    # TODO: Use lightning for loading pretrained models and checkpoints
    total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Total parameters: {total_params:,}")
    if cfg.log:
        wandb.log({"total_parameters": total_params}, commit=False)

    parameter_limit = getattr(cfg, "parameter_limit", None)
    if parameter_limit is not None and total_params > parameter_limit:
        print(
            f"Parameter limit of {parameter_limit:,} exceeded. Skipping this run."
        )
        if cfg.log:
            wandb.log(
                {
                    "parameter_threshold_exceeded": True,
                }
            )
            wandb.run.summary["status"] = "parameter_threshold_exceeded"
        return False


    ###### TRAINER ##########################################################
    # TODO: Add profiler
    # TODO: Consider fast_dev_run, overfit_batches, num_sanity_val_steps for testing and debugging
    # TODO: Consider benchmark mode for benchmarking
    # TODO: Consider deterministic mode for reproducibility
    # TODO: Consider `SlurmCluster` class for building slurm scripts
    # TODO: Consider `DistributedDataParallel` for distributed training NLP on large datasets
    # TODO: Consider HyperOptArgumentParser for hyperparameter optimization
    trainer = instantiate(cfg.trainer)
    if not cfg.log:
        trainer.logger = None
    print(f"Using device: {trainer.accelerator}")
    ############################# task instantiation #############################
    routine_factory = instantiate(cfg.routine)
    routine = routine_factory(model=model)
    trainer.fit(routine, datamodule=data_module)
    trainer.test(routine, datamodule=data_module)

    if cfg.log:
        wandb.finish()


if __name__ == "__main__":
    main()
