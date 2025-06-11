import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

import wandb
from deepreaction.utils import DataSplit, load_model, set_seed
from deepreaction.utils.hydra import safe_instantiate

OmegaConf.register_new_resolver("eval", eval)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    cfg = OmegaConf.create(cfg)
    # config mutable
    OmegaConf.set_struct(cfg, False)

    if getattr(cfg, "seed", None) is not None:
        set_seed(cfg.seed)

    if cfg.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    ##### DATA PIPELINE #########################################################
    data_ingestor = safe_instantiate(cfg.data_ingestor)
    print(f"INFO: Datapipeline instantiated successfully")
    dataframes = data_ingestor()
    print(f"INFO: Datapipeline finished successfully")

    ##### DATA MODULE ###########################################################
    dataset_factory = safe_instantiate(cfg.dataset)
    print(f"INFO: Data module factory instantiated successfully")
    datasets = DataSplit(*map(lambda df: dataset_factory(df), dataframes))
    print(f"INFO: Data modules instantiated successfully")

    ##### DATALOADERS ###########################################################
    dataloader_factory = safe_instantiate(cfg.dataloader)

    train_loader = dataloader_factory(
        dataset=datasets.train,
        shuffle=True,
    )
    val_loader = dataloader_factory(
        dataset=datasets.val,
        shuffle=False,
    )
    test_loader = dataloader_factory(
        dataset=datasets.test,
        shuffle=False,
    )
    print(f"INFO: Dataloaders instantiated successfully")

    ##### UPDATE GLOBAL CONFIG FROM DATASET ATTRIBUTES ##############################
    dataset_properties = cfg.get("runtime_agrs_from_train_dataset_props", [])
    if dataset_properties:
        print("INFO: Updating global config with properties of train dataset:")
        for dataset_property in dataset_properties:
            if hasattr(datasets.train, dataset_property):
                value = getattr(train_loader.dataset, dataset_property)
                OmegaConf.update(cfg, dataset_property, value, merge=True)
            else:
                raise AttributeError(
                    f"Attribute '{dataset_property}' not found on datasets.train."
                )

    OmegaConf.resolve(cfg)
    final_cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    print(f"INFO: Final config:\n{OmegaConf.to_yaml(final_cfg_dict)}")

    ##### INITIALIZE W&B ##########################################################
    run_name = getattr(cfg, "run_name", None)
    if cfg.log:
        wandb.init(
            project=cfg.project_name,
            group=cfg.group_name,
            name=run_name,
            config=final_cfg_dict,
        )
        # TODO: Generalize for datasets w/o support for precomputation
        precompute_time = (
            datasets.train.precompute_time
            + datasets.val.precompute_time
            + datasets.test.precompute_time
        )
        wandb.log({"Precompute_time": precompute_time}, commit=False)

    ##### MODEL ##################################################################
    model = safe_instantiate(cfg.model)
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
        if cfg.log:
            wandb.log(
                {
                    "total_parameters": total_params,
                    "parameter_threshold_exceeded": True,
                }
            )
            wandb.run.summary["status"] = "parameter_threshold_exceeded"
        return False

    if cfg.log:
        wandb.log({"total_parameters": total_params}, commit=False)

    ############################# routine instantiation #############################
    safe_instantiate(
        cfg.routine,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model=model,
        device=device,
    )

    if cfg.log:
        wandb.finish()


if __name__ == "__main__":
    main()
