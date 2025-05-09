from functools import partial
import os

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch_geometric.loader import DataLoader

from deeprxn import dataset
from deeprxn.data_pipeline.data_split import DataSplit
from deeprxn.data_pipeline.data_source.data_source import DataSource
from deeprxn.data_pipeline.representation_factory.graph_representation_factory import GraphRepresentationFactory
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

    ##### SOURCE PIPELINE ########################################################
    # TODO: Instantiate source pipeline as a whole using hydra
    data_source: DataSource = hydra.utils.instantiate(cfg.data_cfg.dataset_cfg.data_source_cfg)
    preprocessing_cfg = getattr(cfg.data_cfg.dataset_cfg, "preprocessing_cfg", {})
    preprocessing_pipeline = nn.Sequential(*[
        hydra.utils.instantiate(config)
        for config in preprocessing_cfg.values()
    ])   

    data = data_source.load()
    dataframes = preprocessing_pipeline.forward(data)       
    print(f"DEBUG: Preprocessing pipeline finished successfully")

    ##### SAMPLE PROCESSING PIPELINE #############################################
    # TODO: Generalize pipeline to non-graph representations
    sample_transform_cfg = getattr(cfg.data_cfg, "sample_transform_cfg", {})
    sample_processing_pipeline = nn.Sequential(*[
        GraphRepresentationFactory(
            preconf_repr=hydra.utils.instantiate(
                cfg.data_cfg.representation_cfg
            )),
        *[
            hydra.utils.instantiate(config)
            for _, config in sample_transform_cfg.items()
        ]
    ])
    print(f"DEBUG: Sample processing pipeline instantiated successfully")

    ##### DATASET PROCESSING PIPELINE ############################################
    dataset_transform_cfg = getattr(cfg.data_cfg, "dataset_transform_cfg", {})
    dataset_processing_pipeline = nn.Sequential(*[
            hydra.utils.instantiate(config)
            for _, config in dataset_transform_cfg.items()
        ]
)
    print(f"DEBUG: Dataset processing pipeline instantiated successfully")

    ##### DATASETS ###############################################################
    dataset_partial = hydra.utils.instantiate(
        cfg.data_cfg.dataset_cfg,
        sample_processing_pipeline=sample_processing_pipeline,
    )
    datasets = DataSplit(
        *map(
            lambda df: dataset_processing_pipeline.forward(dataset_partial(data=df)),
            dataframes
        )
    )
    print(f"DEBUG: Datasets instantiated successfully")

    ##### DATALOADERS ###########################################################
    # TODO: Preconfigure dataloader via hydra and instantiate using factory
    dataloader_partial = partial(
        DataLoader,
        batch_size=cfg.data_cfg.batch_size,
        num_workers=cfg.data_cfg.num_workers,
        pin_memory=True,
        sampler=None,
        generator=torch.Generator().manual_seed(0), # TODO: Do not hardcode seed!
    )

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
    print(f"DEBUG: Dataloaders instantiated successfully")

    ##### INITIALIZE W&B ##########################################################
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
