#!/usr/bin/env python3
"""Generate OOD benchmark configs for every model/split combination."""
from __future__ import annotations

import argparse
import copy
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
# Update these constants if your configs live elsewhere.
# Source configs are expected to be named <model>.yaml. To support other schemes,
SOURCE_MODEL_CONFIG_DIR = REPO_ROOT / "conf" / "saved_configs" / "chemtorch_benchmark" / "optimal_model_configs"
OOD_CONFIG_OUTPUT_DIR = REPO_ROOT / "conf" / "saved_configs" / "chemtorch_benchmark" / "ood_benchmark"
SEED_DIR_TEMPLATE = "seed_${seed}_${now:%Y-%m-%d_%H-%M-%S}"
PREDICTION_BASE = "predictions/chemtorch_paper/rdb7_data_split_benchmark"
CHECKPOINT_BASE = "${trainer.default_root_dir}/chemtorch_paper/rdb7_data_split_benchmark"
GROUP_NAME = "chemtorch_data_split_benchmark"
ENABLE_CHECKPOINTING = False
SAVE_PREDICTIONS_FOR = ["train", "test"]
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

SPLIT_RATIO_DEFAULTS = {
    "train_ratio": TRAIN_RATIO,
    "val_ratio": VAL_RATIO,
    "test_ratio": TEST_RATIO,
}


@dataclass(frozen=True)
class SplitSpec:
    """Describe how to build a config variant for a given split."""

    file_name: str
    prefix: str
    splitter: Mapping[str, Any]


def build_split_specs() -> OrderedDict[str, SplitSpec]:
    """Return the ordered list of supported split configurations."""
    return OrderedDict(
        (
            (
                "random_split",
                SplitSpec(
                    file_name="random_split",
                    prefix="random",
                    splitter={
                        "_target_": "chemtorch.components.data_pipeline.data_splitter.RatioSplitter",
                        **SPLIT_RATIO_DEFAULTS,
                        "save_path": None,
                    },
                ),
            ),
            (
                "reactant_scaffold_split",
                SplitSpec(
                    file_name="reactant_scaffold_split",
                    prefix="reactant_scaffold",
                    splitter={
                        "_target_": "chemtorch.components.data_pipeline.data_splitter.ScaffoldSplitter",
                        **SPLIT_RATIO_DEFAULTS,
                        "split_on": "reactant",
                        "mol_idx": 0,
                        "include_chirality": False,
                        "save_path": None,
                    },
                ),
            ),
            (
                "reaction_core_split",
                SplitSpec(
                    file_name="reaction_core_split",
                    prefix="reaction_core",
                    splitter={
                        "_target_": "chemtorch.components.data_pipeline.data_splitter.ReactionCoreSplitter",
                        **SPLIT_RATIO_DEFAULTS,
                        "save_path": None,
                    },
                ),
            ),
            (
                "size_split_asc",
                SplitSpec(
                    file_name="size_split_asc",
                    prefix="size_asc",
                    splitter={
                        "_target_": "chemtorch.components.data_pipeline.data_splitter.SizeSplitter",
                        **SPLIT_RATIO_DEFAULTS,
                        "sort_order": "ascending",
                        "save_path": None,
                    },
                ),
            ),
            (
                "size_split_desc",
                SplitSpec(
                    file_name="size_split_desc",
                    prefix="size_desc",
                    splitter={
                        "_target_": "chemtorch.components.data_pipeline.data_splitter.SizeSplitter",
                        **SPLIT_RATIO_DEFAULTS,
                        "sort_order": "descending",
                        "save_path": None,
                    },
                ),
            ),
            (
                "target_split_asc",
                SplitSpec(
                    file_name="target_split_asc",
                    prefix="target_asc",
                    splitter={
                        "_target_": "chemtorch.components.data_pipeline.data_splitter.TargetSplitter",
                        **SPLIT_RATIO_DEFAULTS,
                        "sort_order": "ascending",
                        "save_path": None,
                    },
                ),
            ),
            (
                "target_split_desc",
                SplitSpec(
                    file_name="target_split_desc",
                    prefix="target_desc",
                    splitter={
                        "_target_": "chemtorch.components.data_pipeline.data_splitter.TargetSplitter",
                        **SPLIT_RATIO_DEFAULTS,
                        "sort_order": "descending",
                        "save_path": None,
                    },
                ),
            ),
        )
    )


SPLIT_SPECS = build_split_specs()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=
        "Create OOD benchmark configs by cloning source model configs. Update the constants at the top of the script to change directories."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific model config stems to process (e.g., atom_han). Defaults to every *.yaml in the source config directory.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=list(SPLIT_SPECS.keys()),
        help="Subset of split configs to generate. Defaults to all supported splits.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite configs if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without writing files.",
    )
    return parser.parse_args()


def discover_models(config_dir: Path) -> List[str]:
    if not config_dir.is_dir():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")
    return sorted(p.stem for p in config_dir.glob("*.yaml"))


def load_yaml_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML config at {path} is not a mapping")
    return data


def write_yaml_config(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, default_flow_style=False)


def update_prediction_paths(config: Dict[str, Any], model_name: str, prefix: str) -> None:
    predictions_dir = f"{PREDICTION_BASE}/{model_name}/{prefix}/{SEED_DIR_TEMPLATE}"
    config["predictions_save_dir"] = predictions_dir
    trainer = config.get("trainer")
    if isinstance(trainer, dict):
        checkpoint = trainer.get("checkpoint_callback")
        if isinstance(checkpoint, dict):
            checkpoint["dirpath"] = (
                f"{CHECKPOINT_BASE}/{prefix}/{SEED_DIR_TEMPLATE}/checkpoints"
            )


def apply_trainer_overrides(config: Dict[str, Any]) -> None:
    trainer = config.get("trainer")
    if isinstance(trainer, dict):
        trainer["enable_checkpointing"] = ENABLE_CHECKPOINTING


def apply_predictions_override(config: Dict[str, Any]) -> None:
    if SAVE_PREDICTIONS_FOR is not None:
        config["save_predictions_for"] = list(SAVE_PREDICTIONS_FOR)


def set_data_splitter(config: Dict[str, Any], splitter_cfg: Mapping[str, Any]) -> None:
    data_module = config.setdefault("data_module", {})
    pipeline = data_module.setdefault("data_pipeline", {})
    pipeline["data_splitter"] = copy.deepcopy(splitter_cfg)


def build_split_config(
    base_config: Dict[str, Any], model_name: str, spec: SplitSpec
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_config)
    cfg["group_name"] = GROUP_NAME
    cfg["run_name"] = f"{spec.prefix}_{model_name}"
    update_prediction_paths(cfg, model_name, spec.prefix)
    apply_trainer_overrides(cfg)
    apply_predictions_override(cfg)
    set_data_splitter(cfg, spec.splitter)
    return cfg


def ensure_requested(models: Iterable[str], available: Iterable[str]) -> List[str]:
    available_set = set(available)
    missing = [m for m in models if m not in available_set]
    if missing:
        raise ValueError(f"Unknown model(s): {', '.join(missing)}")
    return list(models)


def main() -> None:
    args = parse_args()
    source_dir = SOURCE_MODEL_CONFIG_DIR
    output_dir = OOD_CONFIG_OUTPUT_DIR

    available_models = discover_models(source_dir)
    if not available_models:
        raise RuntimeError("No source model configs found.")

    if args.models:
        model_names = ensure_requested(args.models, available_models)
    else:
        model_names = available_models

    if args.splits:
        split_names = args.splits
    else:
        split_names = list(SPLIT_SPECS.keys())

    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name in model_names:
        source_path = source_dir / f"{model_name}.yaml"
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing source config: {source_path}")
        base_config = load_yaml_config(source_path)
        target_dir = output_dir / model_name
        target_dir.mkdir(parents=True, exist_ok=True)

        for split_name in split_names:
            spec = SPLIT_SPECS[split_name]
            target_path = target_dir / f"{spec.file_name}.yaml"
            if target_path.exists() and not args.overwrite:
                print(
                    f"[SKIP] {target_path} exists. Use --overwrite to regenerate.",
                    file=sys.stderr,
                )
                continue

            new_config = build_split_config(base_config, model_name, spec)
            if args.dry_run:
                print(f"[DRY-RUN] Would write {target_path}")
            else:
                write_yaml_config(target_path, new_config)
                print(f"[WRITE] {target_path}")


if __name__ == "__main__":
    main()
