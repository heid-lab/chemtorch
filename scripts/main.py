import os
from typing import List

import hydra
import torch
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig, OmegaConf

import wandb
from deeprxn.data_pipeline.data_pipeline import DataPipelineComponent, SourcePipeline
from deeprxn.dataset.mol_graph_dataset import construct_loader
from deeprxn.representation.representation_factory import RepresentationFactory
from deeprxn.utils import load_model, set_seed

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
    # TODO: Instantiate data pipeline as a whole using hydra
    pipeline_components: List[DataPipelineComponent] = [
        hydra.utils.instantiate(component_cfg)
        for component_cfg in cfg.data_cfg.dataset_cfg.pipeline_component.values()
    ]
    source_pipeline = SourcePipeline(pipeline_components)

    data_split = source_pipeline.forward()
    print(f"DEBUG: Source pipeline finished successfully")

    representation_factory = RepresentationFactory(
        preconf_repr=hydra.utils.instantiate(cfg.data_cfg.representation_cfg)
    )

    # TODO: Add dataset factory? 
    dataset_partial: Dataset = hydra.utils.instantiate(
        cfg.data_cfg.dataset_cfg,
        representation_factory=representation_factory,
        transform_cfg=getattr(cfg.data_cfg, "transform_cfg", None)  # catches transform_cfg: null
    )

    train_set = dataset_partial(data=data_split.train)
    val_set = dataset_partial(data=data_split.val)
    test_set = dataset_partial(data=data_split.test)

    # TODO: Add dataset wide opertaionts (e.g. data augmentation, or dataset statistics needed for PNA)
    # TODO: Compute endocdings here for whole dataset (not for each batch)

    # TODO: Preconfigure dataloader via hydra and instantiate using factory
    train_loader = construct_loader(
        dataset=train_set,
        batch_size=cfg.data_cfg.batch_size,
        shuffle=True,
        num_workers=cfg.data_cfg.num_workers,
    )
    val_loader = construct_loader(
        dataset=val_set,
        batch_size=cfg.data_cfg.batch_size,
        shuffle=False,
        num_workers=cfg.data_cfg.num_workers,
    )
    test_loader = construct_loader(
        dataset=test_set,
        batch_size=cfg.data_cfg.batch_size,
        shuffle=False,
        num_workers=cfg.data_cfg.num_workers,
    )    

    # TODO: Move this to graph dataset, or even better, to hydra
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
        # TODO: DON'T HARD CODE THIS
        #### for models needing precomputed statistics on the dataset, e.g. PNA
        transform_cfg = getattr(cfg.data_cfg, "transform_cfg", None)
        if transform_cfg and hasattr(
            transform_cfg, "batched_degree_statistics"
        ):
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
