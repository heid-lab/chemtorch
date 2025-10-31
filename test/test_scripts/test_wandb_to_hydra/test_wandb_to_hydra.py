import pytest
import yaml
import tempfile
import shutil
from pathlib import Path
import sys

# Add the project root directory to the path so we can import the module
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from scripts.wandb_to_hydra import preprocess_wandb_config


class TestPreprocessWandbConfig:
    """Test suite for the preprocess_wandb_config function."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.wandb_dir = self.test_dir / "wandb"
        self.wandb_dir.mkdir()
        
        # Sample WandB config
        self.sample_config = {
            "_wandb": {"framework": "pytorch"},
            "desc": "Test",
            "wandb_version": "0.15.0",
            "model": {"value": {"hidden_size": {"value": 128}}},
            "batch_size": {"value": 64}
        }

    def teardown_method(self):
        """Clean up after each test method."""
        shutil.rmtree(self.test_dir)

    def create_run_directory(self, run_id, config_data=None):
        """Helper to create a mock WandB run directory."""
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

    def test_successful_preprocessing(self):
        """Test successful preprocessing with typical WandB config."""
        run_id = "abc123"
        self.create_run_directory(run_id)
        
        output_path = self.test_dir / "output.yaml"
        preprocess_wandb_config(run_id, output_path, self.wandb_dir)
        
        assert output_path.exists()
        
        with open(output_path, "r") as f:
            result = yaml.safe_load(f)
        
        # Check unwrapping worked
        assert result["model"]["hidden_size"] == 128
        assert result["batch_size"] == 64
        # Check filtered keys are removed
        assert "_wandb" not in result
        assert "desc" not in result
        assert "wandb_version" not in result
        # Check hydra config added
        assert "hydra" in result
        assert result["hydra"]["output_subdir"] is None

    def test_wandb_directory_not_found(self):
        """Test error when wandb directory doesn't exist."""
        with pytest.raises(FileNotFoundError, match="WandB directory not found"):
            preprocess_wandb_config("test", self.test_dir / "output.yaml", self.test_dir / "nonexistent")

    def test_run_directory_not_found(self):
        """Test error when run ID doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Run not found"):
            preprocess_wandb_config("nonexistent", self.test_dir / "output.yaml", self.wandb_dir)

    def test_config_file_not_found(self):
        """Test error when config file is missing."""
        run_id = "missing_config"
        run_dir = self.wandb_dir / f"run-20250813_172329-{run_id}"
        run_dir.mkdir()
        (run_dir / "files").mkdir()
        # Don't create config.yaml
        
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            preprocess_wandb_config(run_id, self.test_dir / "output.yaml", self.wandb_dir)

    def test_exact_run_id_matching(self):
        """Test that run IDs must match exactly (not as substring)."""
        self.create_run_directory("abc123")
        self.create_run_directory("abc")
        
        # Should match only "abc", not "abc123"
        output_path = self.test_dir / "output.yaml"
        preprocess_wandb_config("abc", output_path, self.wandb_dir)
        assert output_path.exists()

    def test_empty_config_file(self):
        """Test handling of empty config file."""
        run_id = "empty"
        run_dir = self.create_run_directory(run_id, config_data=None)
        config_path = run_dir / "files" / "config.yaml"
        config_path.write_text("")  # Empty file
        
        output_path = self.test_dir / "output.yaml"
        preprocess_wandb_config(run_id, output_path, self.wandb_dir)
        
        with open(output_path, "r") as f:
            result = yaml.safe_load(f)
        
        # Should still have hydra config
        assert "hydra" in result

    def test_config_with_no_value_wrappers(self):
        """Test config that doesn't use value wrappers."""
        config = {
            "model": {"hidden_size": 128},
            "batch_size": 64
        }
        run_id = "no_wrappers"
        self.create_run_directory(run_id, config_data=config)
        
        output_path = self.test_dir / "output.yaml"
        preprocess_wandb_config(run_id, output_path, self.wandb_dir)
        
        with open(output_path, "r") as f:
            result = yaml.safe_load(f)
        
        assert result["model"]["hidden_size"] == 128
        assert result["batch_size"] == 64

    def test_deep_nesting_with_values(self):
        """Test deeply nested config with multiple value wrappers."""
        config = {
            "level1": {
                "value": {
                    "level2": {
                        "value": {
                            "level3": {
                                "value": {
                                    "final": {"value": 42}
                                }
                            }
                        }
                    }
                }
            }
        }
        run_id = "deep"
        self.create_run_directory(run_id, config_data=config)
        
        output_path = self.test_dir / "output.yaml"
        preprocess_wandb_config(run_id, output_path, self.wandb_dir)
        
        with open(output_path, "r") as f:
            result = yaml.safe_load(f)
        
        assert result["level1"]["level2"]["level3"]["final"] == 42
