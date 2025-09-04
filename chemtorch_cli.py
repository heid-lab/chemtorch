import os
from pathlib import Path
from typing import List, cast

import hydra
import torch
import lightning as L
import wandb
from omegaconf import DictConfig, OmegaConf

from chemtorch.core.data_module import DataModule, Stage
from chemtorch.utils.cli import cli_chemtorch_logo
from chemtorch.utils.hydra import safe_instantiate
from chemtorch.utils.misc import save_predictions

OmegaConf.register_new_resolver("eval", eval)

ROOT_DIR = Path(__file__).parent


def _get_prediction_save_path(dataset_key: str, cfg: DictConfig) -> str:
    """
    Get the appropriate save path for predictions based on configuration and dataset key.
    
    Args:
        dataset_key: The dataset split key ("train", "val", "test", or "predict")
        cfg: The configuration object
    
    Returns:
        str: The file path where predictions should be saved
    
    Logic:
        - If predictions_save_dir is set: use {predictions_save_dir}/{dataset_key}_preds.csv
        - If prediction_save_path is set (single task scenario): use prediction_save_path directly
        - Otherwise: raise error (user must specify one of the two)
    """
    if cfg.predictions_save_dir:
        return f"{cfg.predictions_save_dir}/{dataset_key}_preds.csv"
    elif cfg.prediction_save_path:
        return cfg.prediction_save_path
    else:
        raise ValueError(
            f"Cannot determine save path for {dataset_key} predictions. "
            f"For single tasks, specify 'prediction_save_path'. "
            f"For multiple tasks, specify 'predictions_save_dir' and 'save_predictions_for'."
        )

@hydra.main(version_base=None, config_path="conf", config_name="base")
def main(cfg: DictConfig):
    cli_chemtorch_logo()
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
        for single_dataset_key in stages:
            precompute_time = data_module.get_dataset_property(single_dataset_key, "precompute_time")
            wandb.log(
                {f"{single_dataset_key}_precompute_time": precompute_time},
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

    if "predict" in cfg.tasks and not cfg.prediction_save_path:
        raise ValueError("Set `prediction_save_path` in the config if you want to save the predictions for predict task.")

    ###### INFERENCE AND PREDICTION SAVING #####################################
    # Determine which predictions to save
    save_predictions_for = []
    
    # Validate configuration according to the rules
    single_tasks = [task for task in ["predict", "validate", "test"] if task in cfg.tasks]
    has_fit_or_multiple_tasks = "fit" in cfg.tasks or len(cfg.tasks) > 1
    
    # Handle gracefully: if save_predictions_for specified but no paths, just warn and skip
    skip_prediction_saving = cfg.save_predictions_for and not cfg.predictions_save_dir and not cfg.prediction_save_path
    if skip_prediction_saving:
        print("WARNING: 'save_predictions_for' specified but no 'predictions_save_dir' or 'prediction_save_path' given. Skipping prediction saving.")
    
    elif has_fit_or_multiple_tasks and cfg.prediction_save_path and not cfg.save_predictions_for:
        raise ValueError(
            "For fit tasks or multiple tasks, you must specify 'save_predictions_for' to indicate "
            "which datasets to save predictions for, and use 'predictions_save_dir' instead of 'prediction_save_path'."
        )
    
    elif cfg.save_predictions_for and len(cfg.save_predictions_for) > 1 and cfg.prediction_save_path:
        raise ValueError(
            "Cannot use 'prediction_save_path' when saving predictions for multiple datasets. "
            "Use 'predictions_save_dir' instead."
        )
    
    # Determine save_predictions_for list (only if we're not skipping)
    if not skip_prediction_saving:
        # For single task scenarios (predict, validate, test), use prediction_save_path
        if len(single_tasks) == 1 and len(cfg.tasks) == 1 and not cfg.save_predictions_for and cfg.prediction_save_path:
            single_dataset_key = single_tasks[0] if single_tasks[0] != "validate" else "val"
            save_predictions_for = [single_dataset_key]

        # For multi-task scenarios or explicit save_predictions_for, use that list
        elif cfg.save_predictions_for:
            if "all" in cfg.save_predictions_for:
                if len(cfg.save_predictions_for) > 1:
                    raise ValueError("When using 'all' in save_predictions_for, it should be the only item in the list.")
                save_predictions_for = ["train", "val", "test", "predict"]
            else:
                save_predictions_for = cfg.save_predictions_for
        
    # Save predictions if requested
    if save_predictions_for:
        # Final validation before saving
        if len(save_predictions_for) > 1 and cfg.prediction_save_path:
            raise ValueError(
                f"Cannot save predictions for multiple datasets ({save_predictions_for}) "
                f"to a single file ({cfg.prediction_save_path}). Use 'predictions_save_dir' instead."
            )
        
        if not cfg.predictions_save_dir and not cfg.prediction_save_path:
            raise ValueError(
                "Must specify either 'predictions_save_dir' or 'prediction_save_path' to save predictions."
            )
        
        for single_dataset_key in ["train", "val", "test", "predict"]:
            if single_dataset_key in save_predictions_for:
                print(f"Generating predictions for {single_dataset_key} set...")

                # Cast to Stage type for type safety
                stage_key = cast(Stage, single_dataset_key)
                dataloader = data_module.make_dataloader(stage_key)
                dataset = data_module.get_dataset(stage_key)
                
                preds = trainer.predict(routine, ckpt_path=ckpt_for_inference, dataloaders=dataloader)

                if preds:
                    pred_df = dataset.dataframe.copy()
                    save_path = _get_prediction_save_path(single_dataset_key, cfg)
                    save_predictions(
                        preds=preds,
                        reference_df=pred_df,
                        save_path=save_path,
                        log_func=wandb.log if cfg.log else None,
                        root_dir=ROOT_DIR,
                    )
        
        # Validate that all requested dataset keys are valid
        valid_keys = {"train", "val", "test", "predict", "all"}
        if cfg.save_predictions_for:
            invalid_keys = set(cfg.save_predictions_for) - valid_keys
            if invalid_keys:
                raise ValueError(f"Invalid dataset keys in save_predictions_for: {invalid_keys}. Must be one of {valid_keys}.")

    if cfg.log:
        wandb.finish()


if __name__ == "__main__":
    main()
