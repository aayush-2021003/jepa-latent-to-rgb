"""Small CPU checks for split isolation and image-loss gradient flow."""
from __future__ import annotations

import unittest

import torch

from experiments.jepa_cosmos.download_denseworld import Record, select_three_way_source_disjoint
from experiments.vjepa21_cosmos_single_frame.train_image_adapter import objective


class ThreeWaySplitTest(unittest.TestCase):
    def test_exact_counts_and_disjoint_sources(self):
        records = [Record("remote", "local", f"g{group}/{clip}", f"key{group}/{clip}", f"g{group}")
                   for group in range(140) for clip in range(100)]
        splits = select_three_way_source_disjoint(records, 9000, 500, 500, 20, 42)
        self.assertEqual({key: len(value) for key, value in splits.items()},
                         {"train": 9000, "val": 500, "test": 500})
        sources = {split: {record.source_group for record in rows} for split, rows in splits.items()}
        self.assertFalse(sources["train"] & sources["val"])
        self.assertFalse(sources["train"] & sources["test"])
        self.assertFalse(sources["val"] & sources["test"])
        self.assertGreaterEqual(len(sources["val"]), 20)
        self.assertGreaterEqual(len(sources["test"]), 20)


class ObjectiveTest(unittest.TestCase):
    def test_latent_mse_is_the_only_training_term(self):
        class NoDecode:
            def decode(self, latent):
                raise AssertionError("Latent-MSE training must not decode images")

        adapter = torch.nn.Conv2d(1, 1, 1, bias=False)
        features = torch.ones(2, 1, 4, 4)
        target = torch.zeros(2, 1, 4, 4)
        prediction = adapter(features)
        loss, parts = objective(
            prediction, target, None, NoDecode(), None,
            {"loss": {"objective": "latent_mse"}},
        )
        self.assertEqual(set(parts), {"latent_mse"})
        self.assertTrue(torch.allclose(loss, (prediction.float() - target.float()).square().mean()))
        loss.backward()
        self.assertGreater(float(adapter.weight.grad.abs().sum()), 0)

    def test_rgb_and_perceptual_terms_reach_adapter(self):
        class Decoder:
            def decode(self, latent):
                return latent.repeat(1, 3, 1, 1)

        class Perceptual(torch.nn.Module):
            def forward(self, prediction, target):
                return (prediction - target).abs().mean(dim=(1, 2, 3), keepdim=True)

        adapter = torch.nn.Conv2d(1, 1, 1)
        features = torch.ones(2, 1, 4, 4)
        target = torch.zeros(2, 1, 4, 4)
        rgb = torch.full((2, 3, 4, 4), 127, dtype=torch.uint8)
        config = {"loss": {"latent_l1": 0, "latent_cosine": 0,
                           "rgb_mse": 1, "perceptual": 0.1}}
        loss, parts = objective(adapter(features), target, rgb, Decoder(), Perceptual(), config)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(adapter.weight.grad.abs().sum()), 0)
        self.assertIn("rgb_mse", parts)
        self.assertIn("rgb_lpips", parts)


if __name__ == "__main__":
    unittest.main()
