import yaml
import argparse
from pathlib import Path


def preprocess_wandb_config(run_id, run_name, group_name, project_dir):
    """
    Preprocess a WandB config file to make it compatible with Hydra.
    Removes the `value` wrapper from each key, filters out `_wandb`, and adds/overwrites `run_name` and `group_name`.
    """
    # Validate input parameters
    if run_id is None:
        raise AttributeError("run_id cannot be None")
    
    # Locate the WandB run directory
    wandb_dir = Path(project_dir) / "wandb"
    if not wandb_dir.exists():
        raise FileNotFoundError(f"❌ WandB directory not found at: {wandb_dir}")
    # Find run with exact match for the run_id
    run_dir = next(
        (wandb_dir / d for d in wandb_dir.iterdir() if d.name.endswith(f"-{run_id}")),
        None,
    )
    if run_dir is None:
        raise FileNotFoundError(f"❌ Run not found for run ID: {run_id}")

    config_path = run_dir / "files" / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"❌ Config file not found at: {config_path}")

        # Load the WandB config
    with open(config_path, "r") as f:
        wandb_config = yaml.safe_load(f)

    # Handle empty config files
    if wandb_config is None:
        wandb_config = {}

    def unwrap_values(d):
        """Recursively unwrap `value` fields in the dictionary and filter out `_wandb`."""
        if isinstance(d, dict):
            # Filter out keys to exclude in final hydra config
            for key in ["_wandb", "desc", "wandb_version"]:
                if key in d:
                    d.pop(key)

            # Unwrap value
            if "value" in d and len(d) == 1:
                return unwrap_values(d["value"])
            return {k: unwrap_values(v) for k, v in d.items()}
        return d

    # Unwrap the WandB config
    hydra_config = unwrap_values(wandb_config)

    # Add/overwrite `run_name` and `group_name`
    hydra_config["name"] = run_name
    hydra_config["group_name"] = group_name

    # Add Hydra configuration to prevent creating additional output directories
    hydra_config["hydra"] = {
        "output_subdir": None,
        "run": {
            "dir": "."
        }
    }

    # Save the modified config to the saved_configs directory
    output_dir = Path(project_dir) / "conf" / "saved_configs" / group_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{run_name}.yaml"

    with open(output_path, "w") as f:
        yaml.dump(hydra_config, f, default_flow_style=False)
    
    print(f"✅ Preprocessed config saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="This script parses the config of a WandB run to Hydra " \
    "format and saves it to the location `conf/saved_configs/group/name`.")
    parser.add_argument(
        "--run-id", "-i", required=True, type=str, help="Run ID of the WandB run."
    )
    parser.add_argument(
        "--group", "-g", required=True, type=str, help="Name of the config group, i.e. the folder that the config is saved to."
    )
    parser.add_argument(
        "--name", "-n", required=True, type=str, help="Name for the new resulting hydra config file."
    )
    parser.add_argument(
        "--project-dir", type=str, help="Project directory path. If not provided, uses the parent of the script directory."
    )
    args = parser.parse_args()

    # Determine the project directory
    if args.project_dir:
        project_dir = Path(args.project_dir)
    else:
        # Determine the project directory as the parent of the script directory
        script_dir = Path(__file__).resolve().parent
        project_dir = script_dir.parent

    preprocess_wandb_config(args.run_id, args.name, args.group, project_dir)


if __name__ == "__main__":
    main()