import logging
import os
import logging
from pathlib import Path
from typing import Any, Dict, List, cast, Callable, Optional

import hydra
import pandas as pd
import torch
import lightning as L
import wandb
from omegaconf import DictConfig, OmegaConf

from chemtorch.core.property_system import DatasetProperty, compute_property_with_dataset_handling, resolve_sources
from chemtorch.core.data_module import DataModule
from chemtorch.utils.cli import cli_chemtorch_logo
from chemtorch.utils.hydra import safe_instantiate
from chemtorch.utils.misc import handle_prediction_saving
from chemtorch.utils.types import DatasetKey, PropertySource, RoutineFactoryProtocol

ROOT_DIR = Path(__file__).parent.parent.parent
CONF_DIR = ROOT_DIR / "conf"

OmegaConf.register_new_resolver("eval", eval)

@hydra.main(version_base=None, config_path=str(CONF_DIR), config_name="base")
def main(cfg: DictConfig):
    cli_chemtorch_logo()

    # Configure logging for the entire application
    # Force reconfiguration to override any Hydra logging setup
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s',
        force=True
    )
    
    # Set the root logger level explicitly
    logging.getLogger().setLevel(logging.INFO)
    # Ensure chemtorch loggers propagate to root
    logging.getLogger('chemtorch').setLevel(logging.INFO)
    
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
        if cfg.data_module.data_pipeline.column_mapper.label is not None:
            OmegaConf.update(
                cfg=cfg,
                key="data_module.data_pipeline.column_mapper.label",
                value=None,
                merge=True,
            )

        if cfg.data_module.data_pipeline.data_splitter is not None:
            OmegaConf.update(
                cfg=cfg,
                key="data_module.data_pipeline.data_splitter",
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

    ##### INITIALIZE W&B (BASIC) ##########################################################
    # Initialize wandb first with basic project info, without the full config yet
    if cfg.log:
        wandb.init(
            project=cfg.project_name,
            group=cfg.group_name,
            name=cfg.run_name,
            # Don't pass config yet - we'll update it after computing dataset properties
        )

    ##### RUNTIME DATASET PROPERTIES #########################################
    data_module: DataModule = safe_instantiate(cfg.data_module)
    runtime_props_from_data: Dict[str, DatasetProperty] = cfg.get("props", {})
    logging.debug(runtime_props_from_data)

    # NOTE: Properties cannot be computed on test datasets other than the base "test" dataset
    for property_cfg in runtime_props_from_data.values():
        property: DatasetProperty = safe_instantiate(property_cfg)
        resolved_sources = resolve_sources(cast(PropertySource, property.source), cfg.tasks)
        for source in resolved_sources:
            prop_val = property.compute(data_module.get_dataset(source))
            prop_key = property.name if property.source == "any" else f"{source}_{property.name}"
            if cfg.log and property.log:
                wandb.log({f"{prop_key}": prop_val}, commit=False)
            if property.add_to_cfg and prop_key not in cfg:
                OmegaConf.update(cfg, prop_key, prop_val, merge=True)

    # Resolve the config after all properties have been added and log it
    OmegaConf.resolve(cfg)
    if cfg.log:
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True), allow_val_change=True)
    logging.debug(OmegaConf.to_yaml(cfg))

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

    # Handle conditional checkpointing
    if cfg.trainer.enable_checkpointing and "checkpoint_callback" in cfg.trainer:
        cfg.trainer.callbacks.append(cfg.trainer.checkpoint_callback)

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
    routine_factory: RoutineFactoryProtocol = safe_instantiate(cfg.routine)
    routine: L.LightningModule = routine_factory(model=model, test_dataloader_idx_to_suffix=data_module.maybe_get_test_dataloader_idx_to_suffix())

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
        # Use the datamodule's test_dataloader method which handles both single and multiple test datasets
        trainer.test(routine, datamodule=data_module, ckpt_path=ckpt_for_inference)

    if "predict" in cfg.tasks and not (cfg.predictions_save_path or cfg.predictions_save_dir):
        raise ValueError("Set either `predictions_save_path` or `predictions_save_dir` in the config to save the predictions.")

    ###### INFERENCE AND PREDICTION SAVING #####################################
    # Create closures to encapsulate the prediction generation logic
    def get_preds_func(dataset_key: str) -> List[Any]:
        # dataset_key = cast(DatasetKey, dataset_key)
        # TODO: make_dataloader should return a single dataloader
        dataloader = data_module.make_dataloader(dataset_key)
        preds = trainer.predict(routine, ckpt_path=ckpt_for_inference, dataloaders=dataloader)
        return preds if preds else []
    
    def get_reference_df_func(dataset_key: str) -> pd.DataFrame:
        dataset_key = cast(DatasetKey, dataset_key)
        dataset = data_module.get_dataset(dataset_key)
        return dataset.dataframe.copy()

    handle_prediction_saving(
        get_preds_func=get_preds_func,
        get_reference_df_func=get_reference_df_func,
        get_dataset_names_func=data_module.get_dataset_names,
        predictions_save_dir=cfg.get("predictions_save_dir", None),
        predictions_save_path=cfg.get("predictions_save_path", None),
        save_predictions_for=cfg.get("save_predictions_for", None),
        tasks=cfg.get("tasks", None),
        log_func=wandb.log if cfg.log else None,
        root_dir=ROOT_DIR,
    )


    if cfg.log:
        wandb.finish()


if __name__ == "__main__":
    main()