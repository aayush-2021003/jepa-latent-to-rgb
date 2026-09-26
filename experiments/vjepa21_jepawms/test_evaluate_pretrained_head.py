"""CPU-only geometry and scoring tests for the pretrained-head evaluator."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from experiments.vjepa21_jepawms.evaluate_pretrained_head import (
    decoder_features, save_panel, score, validate_cache,
)


class HeadEvaluationTests(unittest.TestCase):
    def test_feature_geometry(self):
        cached = torch.randn(2, 1408, 1, 24, 24)
        features = decoder_features(cached, grid=16, channels=1408)
        self.assertEqual(tuple(features.shape), (2, 1, 1408, 16, 16))
        with self.assertRaises(ValueError):
            decoder_features(cached[:, :100], grid=16, channels=1408)

    def test_score_identical_images(self):
        image = torch.full((2, 3, 16, 16), 0.5)
        perceptual = lambda prediction, target: (prediction - target).abs().mean(
            dim=(1, 2, 3), keepdim=True
        )
        metrics = score(image, image, perceptual)
        for key in ("l1", "mse", "lpips"):
            self.assertTrue(torch.allclose(metrics[key], torch.zeros(2)))

    def test_cache_contract_and_panel(self):
        metadata = {
            "split": "val", "context_frames": 14, "target_frame_number": 15,
            "contains_jepa_predicted": True, "contains_jepa_target": True,
            "contains_target_rgb": True, "contains_previous_rgb": True,
            "input_dim": 1408,
        }
        cache = SimpleNamespace(manifest={"metadata": metadata, "samples": 500})
        validate_cache(cache, "val", 500)
        with self.assertRaises(RuntimeError):
            validate_cache(cache, "test", 500)
        image = torch.zeros(3, 16, 16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "panel.png"
            save_panel(path, "synthetic", {
                name: image for name in ("context", "ground_truth", "oracle", "predicted")
            })
            self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()
