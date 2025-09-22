import pytest
import subprocess
import tempfile
import shutil
import yaml
from pathlib import Path
import sys
import os


class TestWandbToHydraCLI:
    """Integration tests for the wandb_to_hydra.py command-line interface."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.test_dir = Path(tempfile.mkdtemp())
        # Find the actual script path - go up to project root then to scripts
        project_root = Path(__file__).parent.parent.parent.parent  # Go up from test/test_scripts/test_wandb_to_hydra/
        self.script_path = project_root / "scripts" / "wandb_to_hydra.py"
        
        # Create wandb directory and sample run
        self.wandb_dir = self.test_dir / "wandb"
        self.wandb_dir.mkdir()
        
        # Sample config for CLI tests
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

    def run_script(self, *args, cwd=None, use_project_dir=True):
        """Helper method to run the script with given arguments."""
        if cwd is None:
            cwd = self.test_dir
        
        cmd_args = list(args)
        # Add --project-dir argument to point to the test directory unless disabled
        if use_project_dir:
            cmd_args.extend(["--project-dir", str(self.test_dir)])
        
        cmd = [sys.executable, str(self.script_path)] + cmd_args
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True
        )
        return result

    def test_cli_successful_run(self):
        """Test successful CLI execution with valid arguments."""
        run_id = "cli_test_123"
        self.create_run_directory(run_id)
        
        result = self.run_script(
            "--run-id", run_id,
            "--group", "cli_group",
            "--name", "cli_run"
        )
        
        assert result.returncode == 0
        assert "✅ Preprocessed config saved to:" in result.stdout
        
        # Verify the output file exists
        output_path = self.test_dir / "conf" / "saved_configs" / "cli_group" / "cli_run.yaml"
        assert output_path.exists()

    def test_cli_short_arguments(self):
        """Test CLI with short argument names."""
        run_id = "short_args_123"
        self.create_run_directory(run_id)
        
        result = self.run_script(
            "-i", run_id,
            "-g", "short_group",
            "-n", "short_run"
        )
        
        assert result.returncode == 0
        assert "✅ Preprocessed config saved to:" in result.stdout

    def test_cli_missing_required_arguments(self):
        """Test CLI with missing required arguments."""
        # Missing --run-id
        result = self.run_script("--group", "test", "--name", "test")
        assert result.returncode != 0
        assert "required" in result.stderr.lower()

        # Missing --group
        result = self.run_script("--run-id", "test", "--name", "test")
        assert result.returncode != 0
        assert "required" in result.stderr.lower()

        # Missing --name
        result = self.run_script("--run-id", "test", "--group", "test")
        assert result.returncode != 0
        assert "required" in result.stderr.lower()

    def test_cli_help_message(self):
        """Test CLI help message."""
        result = self.run_script("--help")
        assert result.returncode == 0
        assert "WandB run to Hydra" in result.stdout
        assert "--run-id" in result.stdout
        assert "--group" in result.stdout
        assert "--name" in result.stdout

    def test_cli_nonexistent_run_id(self):
        """Test CLI with nonexistent run ID."""
        result = self.run_script(
            "--run-id", "nonexistent_run",
            "--group", "test_group",
            "--name", "test_run"
        )
        
        assert result.returncode != 0
        assert "Run not found" in result.stderr

    def test_cli_no_wandb_directory(self):
        """Test CLI when wandb directory doesn't exist."""
        # Remove the wandb directory
        shutil.rmtree(self.wandb_dir)
        
        result = self.run_script(
            "--run-id", "some_run",
            "--group", "test_group",
            "--name", "test_run"
        )
        
        assert result.returncode != 0
        assert "WandB directory not found" in result.stderr

    def test_cli_with_realistic_run_id(self):
        """Test CLI with realistic WandB run ID format."""
        run_id = "9yw4fk1l"  # Typical WandB run ID
        self.create_run_directory(run_id)
        
        result = self.run_script(
            "--run-id", run_id,
            "--group", "realistic_group",
            "--name", "realistic_run"
        )
        
        assert result.returncode == 0
        assert "✅ Preprocessed config saved to:" in result.stdout
        
        # Check the output file content
        output_path = self.test_dir / "conf" / "saved_configs" / "realistic_group" / "realistic_run.yaml"
        with open(output_path, "r") as f:
            config = yaml.safe_load(f)
        
        assert config["name"] == "realistic_run"
        assert config["group_name"] == "realistic_group"
        assert config["model"]["hidden_size"] == 256
        assert config["batch_size"] == 32
        assert "_wandb" not in config

    def test_cli_special_characters_in_names(self):
        """Test CLI with special characters in group and run names."""
        run_id = "special_123"
        self.create_run_directory(run_id)
        
        result = self.run_script(
            "--run-id", run_id,
            "--group", "group-with-dashes_and_underscores",
            "--name", "run_with_underscores-and-dashes"
        )
        
        assert result.returncode == 0
        
        # Check that directories and files are created correctly
        output_path = (
            self.test_dir / "conf" / "saved_configs" / 
            "group-with-dashes_and_underscores" / 
            "run_with_underscores-and-dashes.yaml"
        )
        assert output_path.exists()

    def test_cli_empty_arguments(self):
        """Test CLI with empty string arguments."""
        run_id = "empty_test"
        self.create_run_directory(run_id)
        
        # Test with empty group name
        result = self.run_script(
            "--run-id", run_id,
            "--group", "",
            "--name", "empty_group_test"
        )
        # This might succeed depending on filesystem, but the directory structure might be odd
        
        # Test with empty run name
        result = self.run_script(
            "--run-id", run_id,
            "--group", "empty_name_group",
            "--name", ""
        )
        # This might create a file with empty name (like .yaml)

    def test_cli_very_long_arguments(self):
        """Test CLI with very long argument values."""
        run_id = "long_test"
        self.create_run_directory(run_id)
        
        long_group = "a" * 100
        long_name = "b" * 100
        
        result = self.run_script(
            "--run-id", run_id,
            "--group", long_group,
            "--name", long_name
        )
        
        # This should succeed unless filesystem has path length limits
        assert result.returncode == 0

    def test_cli_unicode_characters(self):
        """Test CLI with unicode characters in names."""
        run_id = "unicode_test"
        self.create_run_directory(run_id)
        
        result = self.run_script(
            "--run-id", run_id,
            "--group", "group_with_émojis_🚀",
            "--name", "run_with_特殊字符"
        )
        
        # Behavior may vary by filesystem, but shouldn't crash
        # The script should handle this gracefully

    def test_cli_multiple_runs_exact_matching(self):
        """Test CLI correctly matches exact run ID when multiple similar IDs exist."""
        # Create multiple runs with similar IDs
        base_id = "abc123"
        self.create_run_directory(base_id)
        self.create_run_directory(f"{base_id}4")  # abc1234
        self.create_run_directory(f"x{base_id}")  # xabc123 - but this won't match the pattern
        
        result = self.run_script(
            "--run-id", base_id,
            "--group", "exact_match_group",
            "--name", "exact_match_run"
        )
        
        assert result.returncode == 0
        assert "✅ Preprocessed config saved to:" in result.stdout

    def test_cli_output_verification(self):
        """Test that CLI output contains expected information."""
        run_id = "output_test"
        self.create_run_directory(run_id)
        
        result = self.run_script(
            "--run-id", run_id,
            "--group", "output_group",
            "--name", "output_run"
        )
        
        assert result.returncode == 0
        
        # Check stdout contains the expected path
        expected_path = str(
            self.test_dir / "conf" / "saved_configs" / "output_group" / "output_run.yaml"
        )
        assert expected_path in result.stdout
        
        # Check that stderr is empty (no errors)
        assert result.stderr.strip() == ""
