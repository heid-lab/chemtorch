import os

import hydra
from lightning import Trainer, seed_everything
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import torch
from omegaconf import DictConfig, OmegaConf

import wandb
from deepreaction.data_module_lightning import DataModule
from deepreaction.supervised_routine_lightning import SupervisedRoutine
from deepreaction.utils import DataSplit
from deepreaction.utils import load_model, set_seed
from deepreaction.utils import CallableCompose

OmegaConf.register_new_resolver("eval", eval)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # config mutable
    OmegaConf.set_struct(cfg, False)

    seed = getattr(cfg, "seed", 0)
    seed_everything(seed)

    if getattr(cfg, "seed", None) is not None:
        set_seed(cfg.seed)

    if cfg.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    ##### DATA MODULE #########################################################
    data_pipeline = hydra.utils.instantiate(cfg.data_pipeline)
    dataset_factory = hydra.utils.instantiate(cfg.dataset)
    dataloader_factory = hydra.utils.instantiate(cfg.dataloader)
    data_module = DataModule(
        data_pipeline=data_pipeline,
        dataset_factory=dataset_factory,
        dataloader_factory=dataloader_factory,
    )

    ##### UPDATE GLOBAL CONFIG FROM DATASET ATTRIBUTES ##############################
    dataset_properties = cfg.get("runtime_config_parameters_from_dataset", [])
    if dataset_properties:
        print(
            "INFO: Updating global config with properties of training dataset"
        )
        for dataset_property in dataset_properties:
            OmegaConf.update(
                cfg=cfg, 
                key=dataset_property, 
                val=data_module.get_dataset_property(stage='train', property=dataset_property),
                merge=True
            )

    run_name = getattr(cfg, "run_name", None)
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)

    # print(f"INFO: Final config:\n{OmegaConf.to_yaml(resolved_cfg)}")

    ##### INITIALIZE W&B ##########################################################
    if cfg.wandb:
        wandb.init(
            project=cfg.project_name,
            group=cfg.group_name,
            name=run_name,
            config=resolved_cfg,
        )
        # TODO: Generalize for datasets w/o support for precomputation
        precompute_time = (
            data_module.get_dataset_property(stage='train', property='precompute_time')
            + data_module.get_dataset_property(stage='val', property='precompute_time')
            + data_module.get_dataset_property(stage='test', property='precompute_time')
        )
        wandb.log(
            {"Precompute_time": precompute_time},
            commit=False,
        )


    ##### MODEL ##################################################################
    model = hydra.utils.instantiate(cfg.model)
    model = model.to(device)

    # TODO: Use lightning for loading pretrained models and checkpoints
    if cfg.use_loaded_model:
        if not os.path.exists(cfg.pretrained_path):
            raise ValueError(
                f"Pretrained model not found at {cfg.pretrained_path}"
            )

        model, _, _, _ = load_model(model, None, cfg.pretrained_path)

        try:
            sample_batch = next(iter(data_module.train_dataloader()))
            model(sample_batch.to(device))
        except Exception as e:
            raise ValueError(
                f"Pretrained model incompatible with dataset: {str(e)}"
            )

    total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Total parameters: {total_params:,}")

    parameter_limit = getattr(cfg, "parameter_limit", None)
    if parameter_limit is not None and total_params > parameter_limit:
        print(
            f"Model has {total_params:,} parameters, which exceeds the parameter limit of {parameter_limit:,}. Skipping this run."
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

    ###### TRAINER ##########################################################
    # TODO: Add callbacks (e.g., EarlyStopping, ModelCheckpoint)
    # TODO: Add W&B logger
    # TODO: Add profiler
    # TODO: Consider fast_dev_run, overfit_batches, num_sanity_val_steps for testing and debugging
    # TODO: Consider benchmark mode for benchmarking
    # TODO: Consider deterministic mode for reproducibility
    # TODO: Consider `SlurmCluster` class for building slurm scripts
    # TODO: Consider `DistributedDataParallel` for distributed training NLP on large datasets
    # TODO: Consider HyperOptArgumentParser for hyperparameter optimization

    accelerator = cfg.get("accelerator", 'auto')
    trainer = Trainer(
        accelerator=accelerator,
    )
    print(f"Using device: {trainer.accelerator}")
    ############################# task instantiation #############################
    # TODO: Recursively instantiate routine with hydra
    routine = SupervisedRoutine(
        model=model,
        loss=hydra.utils.instantiate(cfg.task.loss),
        optimizer_factory=hydra.utils.instantiate(cfg.task.optimizer),
        lr_scheduler=hydra.utils.instantiate(cfg.task.scheduler),
        # TODO: Use TorchMetrics and track inside routine
        metrics={
            'rmse': root_mean_squared_error,
            'mae': mean_absolute_error
        }
    )
    trainer.fit(routine, datamodule=data_module)
    trainer.test(routine, datamodule=data_module)

    if cfg.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
