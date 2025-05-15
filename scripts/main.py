import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

import wandb
from deepreaction.data_pipeline.data_source.data_source import DataSource
from deepreaction.data_pipeline.data_split import DataSplit
from deepreaction.misc import load_model, set_seed
from deepreaction.transform.compose import Compose

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

    ##### SOURCE PIPELINE ########################################################
    # TODO: Instantiate source pipeline as a whole using hydra
    data_source: DataSource = hydra.utils.instantiate(
        cfg.data_cfg.dataset_cfg.data_source_cfg
    )
    preprocessing_cfg = getattr(
        cfg.data_cfg.dataset_cfg, "preprocessing_cfg", {}
    )
    preprocessing_pipeline = nn.Sequential(
        *[
            hydra.utils.instantiate(config)
            for config in preprocessing_cfg.values()
        ]
    )

    data = data_source.load()
    dataframes = preprocessing_pipeline.forward(data)
    print(f"INFO: Preprocessing pipeline finished successfully")

    ##### REPRESENTATION #######################################################
    representation_cfg = getattr(cfg.data_cfg, "representation_cfg", {})
    representation = hydra.utils.instantiate(representation_cfg)
    print(f"INFO: Representation instantiated successfully")

    #### TRANSFORM #############################################################
    transform_cfg = getattr(cfg.data_cfg, "transform_cfg", {})
    transforms = [
        hydra.utils.instantiate(config) for _, config in transform_cfg.items()
    ]
    transform = Compose(transforms)
    print(f"INFO: Transform instantiated successfully")

    ##### DATASET ###############################################################
    dataset_partial = hydra.utils.instantiate(
        cfg.data_cfg.dataset_cfg,
        representation=representation,
        transform=transform,
    )

    datasets = DataSplit(*map(lambda df: dataset_partial(df), dataframes))
    print(f"INFO: Datasets instantiated successfully")

    ##### DATASET TRANSFORM ####################################################
    dataset_transform_cfg = getattr(cfg.data_cfg, "dataset_transform_cfg", {})
    dataset_transform = nn.Sequential(
        *[
            hydra.utils.instantiate(config)
            for _, config in dataset_transform_cfg.items()
        ]
    )
    print(f"INFO: Dataset transform instantiated successfully")

    datasets = DataSplit(
        *map(
            lambda ds: dataset_transform.forward(ds),
            datasets,
        )
    )
    print(f"INFO: Dataset transform applied successfully")

    ##### DATALOADERS ###########################################################
    dataloader_partial = hydra.utils.instantiate(cfg.data_cfg.dataloader_cfg)

    train_loader = dataloader_partial(
        dataset=datasets.train,
        shuffle=True,
    )
    val_loader = dataloader_partial(
        dataset=datasets.val,
        shuffle=False,
    )
    test_loader = dataloader_partial(
        dataset=datasets.test,
        shuffle=False,
    )
    print(f"INFO: Dataloaders instantiated successfully")

    ##### UPDATE GLOBAL CONFIG FROM DATASET ATTRIBUTES ##############################
    cfg_updates_spec = cfg.get("update_cfg_from_dataset", {})
    if cfg_updates_spec:
        print(
            "INFO: Updating global config from train dataset attributes:"
        )
        for cfg_path, dataset_attr_name in cfg_updates_spec.items():
            if hasattr(train_loader.dataset, dataset_attr_name):
                value = getattr(train_loader.dataset, dataset_attr_name)
                OmegaConf.update(cfg, cfg_path, value, merge=True)
                print(
                    f"  - Updated cfg.{cfg_path} with dataset.{dataset_attr_name} (value: {value})"
                )
            else:
                raise ValueError(
                    f"Attribute '{dataset_attr_name}' (for cfg path '{cfg_path}') not found on train_loader.dataset."
                )

    run_name = getattr(cfg, "run_name", None)
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)

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
            datasets.train.precompute_time
            + datasets.val.precompute_time
            + datasets.test.precompute_time
        )
        wandb.log(
            {"Precompute_time": precompute_time},
            commit=False,
        )

    print(OmegaConf.to_yaml(resolved_cfg))

    ##### RUNTIME MODEL CONSTRUCTOR ARGUMENT COLLECTION ###########################
    runtime_init_args = {}
    attrs_to_collect_for_init = cfg.get(
        "runtime_model_init_args_from_dataset", {}
    )

    if attrs_to_collect_for_init:
        print(
            f"INFO: Checking train dataset for runtime model __init__ attributes: {attrs_to_collect_for_init}"
        )
        for model_arg, dataset_prop in attrs_to_collect_for_init.items():
            if hasattr(datasets.train, dataset_prop):
                runtime_init_args[model_arg] = getattr(
                    datasets.train, dataset_prop
                )
            else:
                raise ValueError(
                    f"Required dataset property '{dataset_prop}' for model argument '{model_arg}' not found on training dataset."
                )

    ##### MODEL ##################################################################
    model = hydra.utils.instantiate(cfg.model_cfg, **runtime_init_args)
    model = model.to(device)

    if cfg.use_loaded_model:
        if not os.path.exists(cfg.pretrained_path):
            raise ValueError(
                f"Pretrained model not found at {cfg.pretrained_path}"
            )

        model, _, _, _ = load_model(model, None, cfg.pretrained_path)

        try:
            sample_batch = next(iter(train_loader))
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

    ############################# task instantiation #############################
    hydra.utils.instantiate(
        cfg.task_cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model=model,
        device=device,
    )
    # train()

    if cfg.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
