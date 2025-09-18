import logging
import os
from pathlib import Path
from typing import Any, List, cast, Callable, Optional

import hydra
import pandas as pd
import torch
import lightning as L
import wandb
from omegaconf import DictConfig, OmegaConf

from chemtorch.core.data_module import DataModule, Stage
from chemtorch.utils.cli import cli_chemtorch_logo
from chemtorch.utils.hydra import safe_instantiate
from chemtorch.utils.misc import handle_prediction_saving

ROOT_DIR = Path(__file__).parent

OmegaConf.register_new_resolver("eval", eval)

@hydra.main(version_base=None, config_path="conf", config_name="base")
def main(cfg: DictConfig):
    cli_chemtorch_logo()

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    # config mutable
    OmegaConf.set_struct(cfg, False)

    seed = getattr(cfg, "seed", 0)
    L.seed_everything(seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = (":4096:8")     # https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility

    ##### DATA MODULE ##############################################################
    if "predict" in cfg.tasks:
        # Remove label from column mapper since it should not be specified for inference (error will be
        # raised if it is specified)
        if cfg.data_pipeline.column_mapper.label is not None:
            OmegaConf.update(
                cfg=cfg,
                key="data_pipeline.column_mapper.label",
                value=None,
                merge=True,
            )
        # Set the standardizer of regression routine to None since it should be loaded from the checkpoint
        # and not created anew for inference.
        if cfg.routine.standardizer is not None:
            OmegaConf.update(
                cfg=cfg,
                key="routine.standardizer",
                value=None,
                merge=True,
            )

    data_pipeline = safe_instantiate(cfg.data_pipeline)
    dataset_factory = safe_instantiate(cfg.dataset)
    dataloader_factory = safe_instantiate(cfg.dataloader)
    data_module = DataModule(
        data_pipeline=data_pipeline,
        dataset_factory=dataset_factory,
        dataloader_factory=dataloader_factory,
    )

    ##### UPDATE GLOBAL CONFIG FROM DATASET ATTRIBUTES ##############################
    # TODO: Bad practice, find a proper solution to pass these dataset
    # properties to the model/routine
    if "fit" in cfg.tasks:
        key = "train"
    elif "validate" in cfg.tasks:
        key = "val"
    elif "test" in cfg.tasks:
        key = "test"
    else:
        key = "predict"
    dataset_properties = cfg.get("runtime_args_from_dataset", [])
    if dataset_properties:
        for dataset_property in dataset_properties:
            OmegaConf.update(
                cfg=cfg,
                key=dataset_property,
                value=data_module.get_dataset_property(
                    key=key, property=dataset_property
                ),
                merge=True,
            )
    
    run_name = getattr(cfg, "run_name", None)
    OmegaConf.resolve(cfg)
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)

    # print(f"INFO: Final config:\n{OmegaConf.to_yaml(resolved_cfg)}")

    ##### INITIALIZE W&B ##########################################################
    if cfg.log:
        wandb.init(
            project=cfg.project_name,
            group=cfg.group_name,
            name=run_name,
            config=resolved_cfg,  # type: ignore
        )
        stages: List[Stage] = []
        if "fit" in cfg.tasks:
            stages.append("train")
        if "validate" in cfg.tasks:
            stages.append("val")
        if "test" in cfg.tasks:
            stages.append("test")
        if "predict" in cfg.tasks:
            stages.append("predict")
        for dataset_key in stages:
            precompute_time = data_module.get_dataset_property(dataset_key, "precompute_time")
            wandb.log(
                {f"{dataset_key}_precompute_time": precompute_time},
                commit=False,
            )

    ###### TRAINER ##########################################################
    # TODO: Add profiler
    # TODO: Consider fast_dev_run, overfit_batches, num_sanity_val_steps for testing and debugging
    # TODO: Consider benchmark mode for benchmarking
    # TODO: Consider deterministic mode for reproducibility
    # TODO: Consider `SlurmCluster` class for building slurm scripts
    # TODO: Consider `DistributedDataParallel` for distributed training NLP on large datasets
    # TODO: Consider HyperOptArgumentParser for hyperparameter optimization

    # MPS is not fully supported yet and can cause issues with certain operations
    # https://lightning.ai/docs/pytorch/stable/accelerators/mps_basic.html
    if torch.backends.mps.is_available() and cfg.trainer.accelerator == "auto":
        cfg.trainer.accelerator = "cpu"

    trainer: L.Trainer = safe_instantiate(cfg.trainer)
    if not cfg.log:
        trainer.logger = None
    # print(f"Using device: {trainer.accelerator}")

    ##### MODEL ##################################################################
    model: torch.nn.Module = safe_instantiate(cfg.model)
    # print(model)
    # print(cfg.model)
    # TODO: Use lightning for loading pretrained models and checkpoints
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Total parameters: {total_params:,}")
    if cfg.log:
        wandb.log({"total_parameters": total_params}, commit=False)

    parameter_limit = getattr(cfg, "parameter_limit", None)
    if parameter_limit is not None and total_params > parameter_limit:
        # print(f"Parameter limit of {parameter_limit:,} exceeded. Skipping this run.")
        if cfg.log:
            wandb.log(
                {
                    "parameter_threshold_exceeded": True,
                }
            )
            wandb.run.summary["status"] = "parameter_threshold_exceeded"  # type: ignore
        return False

    ###### ROUTINE ##########################################################
    routine_factory = safe_instantiate(cfg.routine)
    routine: L.LightningModule = routine_factory(model=model)

    # load model checkpoint if specified
    ckpt_path = None
    if cfg.load_model:
        if cfg.ckpt_path is None:
            raise ValueError("ckpt_path must be provided when load_model is True.")
        ckpt_path = cfg.ckpt_path

    ckpt_for_inference = ckpt_path

    ###### EXECUTION ##########################################################
    if "fit" in cfg.tasks:
        trainer.fit(routine, datamodule=data_module, ckpt_path=ckpt_path)
        # If the model is continued to be trained from the given checkpoint then use the latest
        # model after trainer.fit() for validate, test, predict by passing ckpt_path=None
        ckpt_for_inference = None

    if "validate" in cfg.tasks:
        trainer.validate(routine, datamodule=data_module, ckpt_path=ckpt_for_inference)

    if "test" in cfg.tasks:
        trainer.test(routine, datamodule=data_module, ckpt_path=ckpt_for_inference)

    if "predict" in cfg.tasks and not cfg.predictions_save_path:
        raise ValueError("Set `predictions_save_path` in the config to save the predictions.")

    ###### INFERENCE AND PREDICTION SAVING #####################################
    # Create closures to encapsulate the prediction generation logic
    def get_preds_func(dataset_key: str) -> List[Any]:
        stage_key = cast(Stage, dataset_key)
        dataloader = data_module.make_dataloader(stage_key)
        preds = trainer.predict(routine, ckpt_path=ckpt_for_inference, dataloaders=dataloader)
        return preds if preds else []
    
    def get_reference_df_func(dataset_key: str) -> pd.DataFrame:
        stage_key = cast(Stage, dataset_key)
        dataset = data_module.get_dataset(stage_key)
        return dataset.dataframe.copy()

    handle_prediction_saving(
        get_preds_func=get_preds_func,
        get_reference_df_func=get_reference_df_func,
        predictions_save_dir=cfg.predictions_save_dir,
        predictions_save_path=cfg.predictions_save_path,
        save_predictions_for=cfg.save_predictions_for,
        tasks=cfg.tasks,
        log_func=wandb.log if cfg.log else None,
        root_dir=ROOT_DIR,
    )


    if cfg.log:
        wandb.finish()


if __name__ == "__main__":
    main()
