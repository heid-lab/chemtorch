import pytest
import subprocess
import tempfile
import shutil
import yaml
from pathlib import Path
import sys

# Add the project root directory to the path so we can import the module
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from scripts.wandb_to_hydra import get_parser


class TestWandbToHydraCLI:
    """Integration tests for the wandb_to_hydra.py command-line interface."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.test_dir = Path(tempfile.mkdtemp())
        project_root = Path(__file__).parent.parent.parent.parent
        self.script_path = project_root / "scripts" / "wandb_to_hydra.py"
        
        self.wandb_dir = self.test_dir / "wandb"
        self.wandb_dir.mkdir()
        
        self.sample_config = {
            "_wandb": {"framework": "pytorch"},
            "model": {"value": {"hidden_size": {"value": 256}}},
            "batch_size": {"value": 32}
        }

    def teardown_method(self):
        """Clean up after each test method."""
        shutil.rmtree(self.test_dir)

    def create_run_directory(self, run_id, config_data=None):
        """Helper method to create a mock WandB run directory."""
        if config_data is None:
            config_data = self.sample_config
            
        run_dir = self.wandb_dir / f"run-20250813_172329-{run_id}"
        run_dir.mkdir()
        files_dir = run_dir / "files"
        files_dir.mkdir()
        
        config_path = files_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
        
        return run_dir

    def run_script(self, *args):
        """Helper method to run the script with given arguments."""
        cmd = [sys.executable, str(self.script_path)] + list(args)
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result

    def test_cli_successful_run(self):
        """Test successful CLI execution with valid arguments."""
        run_id = "abc123"
        self.create_run_directory(run_id)
        
        output_path = self.test_dir / "output.yaml"
        result = self.run_script(
            "--run-id", run_id,
            "--output-path", str(output_path),
            "--wandb-dir", str(self.wandb_dir)
        )
        
        assert result.returncode == 0
        assert output_path.exists()
        assert "✅ Preprocessed config saved" in result.stdout

    def test_cli_short_arguments(self):
        """Test CLI with short argument names."""
        run_id = "short123"
        self.create_run_directory(run_id)
        
        output_path = self.test_dir / "output.yaml"
        result = self.run_script(
            "-i", run_id,
            "-o", str(output_path),
            "-d", str(self.wandb_dir)
        )
        
        assert result.returncode == 0
        assert output_path.exists()

    def test_cli_missing_required_arguments(self):
        """Test CLI fails gracefully when required arguments are missing."""
        result = self.run_script("--run-id", "test")
        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "error" in result.stderr.lower()

    def test_cli_help_message(self):
        """Test that --help works and shows useful information."""
        result = self.run_script("--help")
        assert result.returncode == 0
        assert "run-id" in result.stdout
        assert "output-path" in result.stdout
        assert "wandb-dir" in result.stdout

    def test_cli_nonexistent_run_id(self):
        """Test CLI handles nonexistent run ID gracefully."""
        output_path = self.test_dir / "output.yaml"
        result = self.run_script(
            "--run-id", "nonexistent",
            "--output-path", str(output_path),
            "--wandb-dir", str(self.wandb_dir)
        )
        
        assert result.returncode != 0
        assert "Run not found" in result.stderr

    def test_cli_nonexistent_wandb_directory(self):
        """Test CLI handles nonexistent wandb directory gracefully."""
        output_path = self.test_dir / "output.yaml"
        result = self.run_script(
            "--run-id", "test",
            "--output-path", str(output_path),
            "--wandb-dir", str(self.test_dir / "nonexistent")
        )
        
        assert result.returncode != 0
        assert "WandB directory not found" in result.stderr

    def test_cli_relative_output_path(self):
        """Test CLI with relative output path."""
        run_id = "rel123"
        self.create_run_directory(run_id)
        
        # Use relative path for output
        result = self.run_script(
            "--run-id", run_id,
            "--output-path", "test_output.yaml",
            "--wandb-dir", str(self.wandb_dir)
        )
        
        assert result.returncode == 0
        # Output should be in project directory
        project_root = Path(__file__).parent.parent.parent.parent
        output_path = project_root / "test_output.yaml"
        assert output_path.exists()
        # Clean up
        output_path.unlink()

    def test_get_parser_function(self):
        """Test that get_parser returns a valid ArgumentParser."""
        parser = get_parser()
        assert parser is not None
        
        # Test that it can parse valid arguments
        args = parser.parse_args([
            "--run-id", "test",
            "--output-path", "output.yaml",
            "--wandb-dir", "wandb"
        ])
        
        assert args.run_id == "test"
        assert args.output_path == "output.yaml"
        assert args.wandb_dir == "wandb"
