"""Offline checks for permanent Hugging Face epoch checkpoint names."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.vjepa21_cosmos_single_frame.tracking import push_checkpoint_to_hub


class EpochCheckpointUploadTest(unittest.TestCase):
    def test_numbered_epoch_checkpoint_and_metrics_are_uploaded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "adapter_epoch_001.pt"
            checkpoint.write_bytes(b"checkpoint")
            config_path = root / "config.yaml"
            config_path.write_text("experiment: test\n")
            config = {
                "_config_path": str(config_path),
                "huggingface": {"repo_id": "owner/adapter", "private": True},
                "loss": {"objective": "latent_mse"},
                "vjepa21": {"checkpoint_url": "https://example.org/vjepa.pt"},
                "cosmos": {"model_id": "cosmos-test"},
            }
            with patch(
                "experiments.vjepa21_cosmos_single_frame.tracking.require_secret",
                return_value="test-token",
            ), patch("huggingface_hub.HfApi") as hub_class:
                url = push_checkpoint_to_hub(config, checkpoint, {"epoch": 1}, "epoch_001")
            paths = {
                call.kwargs["path_in_repo"]
                for call in hub_class.return_value.upload_file.call_args_list
            }
            self.assertIn("checkpoints/epoch_001.pt", paths)
            self.assertIn("metrics/epoch_001.json", paths)
            self.assertEqual(
                url,
                "https://huggingface.co/owner/adapter/blob/main/checkpoints/epoch_001.pt",
            )


if __name__ == "__main__":
    unittest.main()
