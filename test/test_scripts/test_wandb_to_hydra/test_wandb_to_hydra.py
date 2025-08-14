import pytest
import yaml
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch
import sys

# Add the scripts directory to the path so we can import the module
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from scripts.wandb_to_hydra import preprocess_wandb_config


class TestPreprocessWandbConfig:
    """Test suite for the preprocess_wandb_config function."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory structure for testing
        self.test_dir = Path(tempfile.mkdtemp())
        self.wandb_dir = self.test_dir / "wandb"
        self.wandb_dir.mkdir()
        
        # Sample WandB config with nested values and unwanted keys
        self.sample_wandb_config = {
            "_wandb": {"framework": "pytorch"},
            "desc": "Test description",
            "wandb_version": "0.15.0",
            "model": {
                "value": {
                    "hidden_size": {"value": 128},
                    "dropout": {"value": 0.1},
                    "activation": {"value": "relu"}
                }
            },
            "optimizer": {
                "value": {
                    "lr": {"value": 0.001},
                    "weight_decay": {"value": 0.0001}
                }
            },
            "batch_size": {"value": 64},
            "epochs": {"value": 100}
        }
        
        # Expected output after processing
        self.expected_hydra_config = {
            "model": {
                "hidden_size": 128,
                "dropout": 0.1,
                "activation": "relu"
            },
            "optimizer": {
                "lr": 0.001,
                "weight_decay": 0.0001
            },
            "batch_size": 64,
            "epochs": 100,
            "name": "test_run",
            "group_name": "test_group"
        }

    def teardown_method(self):
        """Clean up after each test method."""
        shutil.rmtree(self.test_dir)

    def create_run_directory(self, run_id, config_data=None):
        """Helper method to create a mock WandB run directory with config."""
        if config_data is None:
            config_data = self.sample_wandb_config
            
        run_dir = self.wandb_dir / f"run-20250813_172329-{run_id}"
        run_dir.mkdir()
        files_dir = run_dir / "files"
        files_dir.mkdir()
        
        config_path = files_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
        
        return run_dir

    def test_successful_preprocessing_realistic_run_id(self):
        """Test successful preprocessing with a realistic run ID."""
        run_id = "9yw4fk1l"
        self.create_run_directory(run_id)
        
        preprocess_wandb_config(run_id, "test_run", "test_group", str(self.test_dir))
        
        # Check that the output file was created
        output_path = self.test_dir / "conf" / "saved_configs" / "test_group" / "test_run.yaml"
        assert output_path.exists()
        
        # Check the content
        with open(output_path, "r") as f:
            result = yaml.safe_load(f)
        
        assert result == self.expected_hydra_config

    def test_successful_preprocessing_short_run_id(self):
        """Test successful preprocessing with a short run ID."""
        run_id = "abc123"
        self.create_run_directory(run_id)
        
        preprocess_wandb_config(run_id, "short_run", "test_group", str(self.test_dir))
        
        output_path = self.test_dir / "conf" / "saved_configs" / "test_group" / "short_run.yaml"
        assert output_path.exists()

    def test_successful_preprocessing_long_run_id(self):
        """Test successful preprocessing with a long run ID."""
        run_id = "very-long-run-id-123456789"
        self.create_run_directory(run_id)
        
        preprocess_wandb_config(run_id, "long_run", "test_group", str(self.test_dir))
        
        output_path = self.test_dir / "conf" / "saved_configs" / "test_group" / "long_run.yaml"
        assert output_path.exists()

    def test_wandb_directory_not_found(self):
        """Test error when wandb directory doesn't exist."""
        # Don't create wandb directory
        shutil.rmtree(self.wandb_dir)
        
        with pytest.raises(FileNotFoundError, match="WandB directory not found"):
            preprocess_wandb_config("9yw4fk1l", "test_run", "test_group", str(self.test_dir))

    def test_run_directory_not_found(self):
        """Test error when run directory doesn't exist."""
        # Create wandb directory but no run directories
        
        with pytest.raises(FileNotFoundError, match="Run not found for run ID"):
            preprocess_wandb_config("nonexistent", "test_run", "test_group", str(self.test_dir))

    def test_config_file_not_found(self):
        """Test error when config.yaml file doesn't exist."""
        run_id = "9yw4fk1l"
        run_dir = self.wandb_dir / f"run-20250813_172329-{run_id}"
        run_dir.mkdir()
        files_dir = run_dir / "files"
        files_dir.mkdir()
        # Don't create config.yaml
        
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            preprocess_wandb_config(run_id, "test_run", "test_group", str(self.test_dir))

    def test_exact_run_id_matching(self):
        """Test that only exact run ID matches are found."""
        # Create runs with similar IDs
        self.create_run_directory("abc123")
        self.create_run_directory("abc1234")  # Similar but different
        self.create_run_directory("xabc123")  # Contains the ID but doesn't end with it
        
        # Should find exact match
        preprocess_wandb_config("abc123", "exact_match", "test_group", str(self.test_dir))
        
        output_path = self.test_dir / "conf" / "saved_configs" / "test_group" / "exact_match.yaml"
        assert output_path.exists()

    def test_ambiguous_run_id_prevention(self):
        """Test that ambiguous run IDs don't cause issues."""
        # Create a run where the ID appears in the middle
        middle_run = self.wandb_dir / "run-20250813_abc123-172329"
        middle_run.mkdir()
        
        # Create the correct run
        self.create_run_directory("abc123")
        
        # Should find the correct run (ending with -abc123)
        preprocess_wandb_config("abc123", "correct_match", "test_group", str(self.test_dir))
        
        output_path = self.test_dir / "conf" / "saved_configs" / "test_group" / "correct_match.yaml"
        assert output_path.exists()

    def test_special_characters_in_run_id(self):
        """Test handling of special characters in run ID."""
        run_id = "test-run_123"
        self.create_run_directory(run_id)
        
        preprocess_wandb_config(run_id, "special_chars", "test_group", str(self.test_dir))
        
        output_path = self.test_dir / "conf" / "saved_configs" / "test_group" / "special_chars.yaml"
        assert output_path.exists()

    def test_empty_run_id(self):
        """Test handling of empty run ID."""
        with pytest.raises(FileNotFoundError, match="Run not found for run ID"):
            preprocess_wandb_config("", "empty_run", "test_group", str(self.test_dir))

    def test_none_run_id(self):
        """Test handling of None run ID."""
        with pytest.raises(AttributeError):
            preprocess_wandb_config(None, "none_run", "test_group", str(self.test_dir))

    def test_numeric_run_id(self):
        """Test handling of numeric run ID."""
        run_id = "123456"
        self.create_run_directory(run_id)
        
        preprocess_wandb_config(run_id, "numeric_run", "test_group", str(self.test_dir))
        
        output_path = self.test_dir / "conf" / "saved_configs" / "test_group" / "numeric_run.yaml"
        assert output_path.exists()

    def test_config_with_no_value_wrappers(self):
        """Test config that doesn't have value wrappers."""
        simple_config = {
            "_wandb": {"framework": "pytorch"},
            "model": {
                "hidden_size": 128,
                "dropout": 0.1
            },
            "batch_size": 64
        }
        
        run_id = "simple123"
        self.create_run_directory(run_id, simple_config)
        
        preprocess_wandb_config(run_id, "simple_run", "test_group", str(self.test_dir))
        
        output_path = self.test_dir / "conf" / "saved_configs" / "test_group" / "simple_run.yaml"
        assert output_path.exists()
        
        with open(output_path, "r") as f:
            result = yaml.safe_load(f)
        
        # Should still filter out _wandb and add name/group_name
        expected = {
            "model": {
                "hidden_size": 128,
                "dropout": 0.1
            },
            "batch_size": 64,
            "name": "simple_run",
            "group_name": "test_group"
        }
        assert result == expected

    def test_config_with_mixed_value_wrappers(self):
        """Test config with some values wrapped and some not."""
        mixed_config = {
            "_wandb": {"framework": "pytorch"},
            "model": {
                "value": {
                    "hidden_size": {"value": 128},
                    "dropout": 0.1  # Not wrapped
                }
            },
            "batch_size": 64,  # Not wrapped
            "optimizer": {"value": {"lr": {"value": 0.001}}}
        }
        
        run_id = "mixed123"
        self.create_run_directory(run_id, mixed_config)
        
        preprocess_wandb_config(run_id, "mixed_run", "test_group", str(self.test_dir))
        
        output_path = self.test_dir / "conf" / "saved_configs" / "test_group" / "mixed_run.yaml"
        assert output_path.exists()
        
        with open(output_path, "r") as f:
            result = yaml.safe_load(f)
        
        expected = {
            "model": {
                "hidden_size": 128,
                "dropout": 0.1
            },
            "batch_size": 64,
            "optimizer": {"lr": 0.001},
            "name": "mixed_run",
            "group_name": "test_group"
        }
        assert result == expected

    def test_empty_config_file(self):
        """Test handling of empty config file."""
        run_id = "empty123"
        run_dir = self.wandb_dir / f"run-20250813_172329-{run_id}"
        run_dir.mkdir()
        files_dir = run_dir / "files"
        files_dir.mkdir()
        
        config_path = files_dir / "config.yaml"
        with open(config_path, "w") as f:
            f.write("")  # Empty file
        
        preprocess_wandb_config(run_id, "empty_config", "test_group", str(self.test_dir))
        
        output_path = self.test_dir / "conf" / "saved_configs" / "test_group" / "empty_config.yaml"
        assert output_path.exists()
        
        with open(output_path, "r") as f:
            result = yaml.safe_load(f)
        
        # Should just have name and group_name
        expected = {
            "name": "empty_config",
            "group_name": "test_group"
        }
        assert result == expected

    def test_malformed_yaml_config(self):
        """Test handling of malformed YAML config."""
        run_id = "malformed123"
        run_dir = self.wandb_dir / f"run-20250813_172329-{run_id}"
        run_dir.mkdir()
        files_dir = run_dir / "files"
        files_dir.mkdir()
        
        config_path = files_dir / "config.yaml"
        with open(config_path, "w") as f:
            f.write("invalid: yaml: content: [unclosed")
        
        with pytest.raises(yaml.YAMLError):
            preprocess_wandb_config(run_id, "malformed_run", "test_group", str(self.test_dir))

    def test_deep_nesting_with_values(self):
        """Test handling of deeply nested config with value wrappers."""
        deep_config = {
            "_wandb": {"framework": "pytorch"},
            "level1": {
                "value": {
                    "level2": {
                        "value": {
                            "level3": {
                                "value": {
                                    "deep_param": {"value": "deep_value"}
                                }
                            }
                        }
                    }
                }
            }
        }
        
        run_id = "deep123"
        self.create_run_directory(run_id, deep_config)
        
        preprocess_wandb_config(run_id, "deep_run", "test_group", str(self.test_dir))
        
        output_path = self.test_dir / "conf" / "saved_configs" / "test_group" / "deep_run.yaml"
        assert output_path.exists()
        
        with open(output_path, "r") as f:
            result = yaml.safe_load(f)
        
        expected = {
            "level1": {
                "level2": {
                    "level3": {
                        "deep_param": "deep_value"
                    }
                }
            },
            "name": "deep_run",
            "group_name": "test_group"
        }
        assert result == expected

    def test_special_characters_in_paths(self):
        """Test handling of special characters in run name and group name."""
        run_id = "special123"
        self.create_run_directory(run_id)
        
        # Test with special characters that should be handled gracefully
        preprocess_wandb_config(run_id, "run-with-dashes_and_underscores", "group_with_underscores", str(self.test_dir))
        
        output_path = self.test_dir / "conf" / "saved_configs" / "group_with_underscores" / "run-with-dashes_and_underscores.yaml"
        assert output_path.exists()

    def test_overwrite_existing_config(self):
        """Test overwriting an existing config file."""
        run_id = "overwrite123"
        self.create_run_directory(run_id)
        
        # Create the output directory and file first
        output_dir = self.test_dir / "conf" / "saved_configs" / "test_group"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "overwrite_run.yaml"
        
        # Write initial content
        with open(output_path, "w") as f:
            yaml.dump({"old": "content"}, f)
        
        # Process the config (should overwrite)
        preprocess_wandb_config(run_id, "overwrite_run", "test_group", str(self.test_dir))
        
        # Check that it was overwritten
        with open(output_path, "r") as f:
            result = yaml.safe_load(f)
        
        assert "old" not in result
        assert result["name"] == "overwrite_run"
        assert result["group_name"] == "test_group"

    @patch('builtins.print')
    def test_success_message_printed(self, mock_print):
        """Test that success message is printed."""
        run_id = "success123"
        self.create_run_directory(run_id)
        
        preprocess_wandb_config(run_id, "success_run", "test_group", str(self.test_dir))
        
        # Check that print was called with success message
        mock_print.assert_called_once()
        args = mock_print.call_args[0][0]
        assert "✅ Preprocessed config saved to:" in args
        assert "success_run.yaml" in args
