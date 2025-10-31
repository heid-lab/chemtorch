import yaml
import argparse
from pathlib import Path


def preprocess_wandb_config(run_id: str, output_path: Path, wandb_dir: Path):
    """
    Preprocess a WandB config file to make it compatible with Hydra.
    Removes the `value` wrapper from each key, filters out `_wandb`.
    """
    # Validate input parameters
    if run_id is None:
        raise AttributeError("run_id cannot be None")
    
    # Locate the WandB run directory
    if not wandb_dir.exists():
        raise FileNotFoundError(f"❌ WandB directory not found: {wandb_dir}")
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

    # Add Hydra configuration to prevent creating additional output directories
    hydra_config["hydra"] = {
        "output_subdir": None,
        "run": {
            "dir": "."
        }
    }

    # Save the modified config to the saved_configs directory
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(hydra_config, f, default_flow_style=False)
    
    print(f"✅ Preprocessed config saved to: {output_path}")


def get_parser() -> argparse.ArgumentParser:
    """Return an argparse.ArgumentParser configured for this script.

    Having this as a separate function makes it trivial for Sphinx extensions
    (or tests) to import and render the current CLI signature automatically.
    """
    parser = argparse.ArgumentParser(
        description=(
            "This script parses the config of a WandB run to Hydra "
            "format, removing the `value` wrappers and saving the result "
            "to a specified output path."
        )
    )
    parser.add_argument(
        "--run-id", "-i", required=True, type=str, help="Run ID of the WandB run."
    )
    parser.add_argument(
        "--output-path", "-o", required=True, type=str, 
        help="Path to save the resulting hydra config file. If not absolute, " \
        "it is considered relative to the project directory (parent of the script directory)."
    )
    parser.add_argument(
        "--wandb-dir", "-d",
        type=str,
        help="Path to the `wandb/` directory. If not provided, the script will search in the " \
        "project directory (parent of the script directory).",
    )
    return parser


def main(argv=None):
    """CLI entry point.

    Accepts an optional `argv` for easier testing; otherwise uses sys.argv.
    """
    parser = get_parser()
    args = parser.parse_args(argv)

    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent

    # Determine the wandb directory
    if args.wandb_dir:
        wandb_dir = Path(args.wandb_dir)
        if not wandb_dir.is_absolute():
            wandb_dir = project_dir / wandb_dir
    else:
        wandb_dir = project_dir / "wandb"

    # Determine the output path
    output_path = Path(args.output_path)
    if not output_path.is_absolute():
        output_path = project_dir / output_path

    preprocess_wandb_config(args.run_id, output_path, wandb_dir)


if __name__ == "__main__":
    main()