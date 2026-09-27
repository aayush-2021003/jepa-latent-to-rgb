"""Small CPU checks for split isolation and image-loss gradient flow."""
from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from copy import deepcopy

import torch

from experiments.jepa_cosmos.download_denseworld import Record, select_three_way_source_disjoint
from experiments.vjepa21_cosmos_single_frame.train_image_adapter import (
    load_adapter_initialization, objective,
)
from experiments.vjepa21_cosmos_single_frame.common import load_config


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

    def test_mse_rgb_lpips_objective_has_expected_terms_and_gradients(self):
        class Decoder:
            def decode(self, latent):
                return latent.repeat(1, 3, 1, 1)

        class Perceptual(torch.nn.Module):
            def forward(self, prediction, target):
                return (prediction - target).abs().mean(dim=(1, 2, 3), keepdim=True)

        adapter = torch.nn.Conv2d(1, 1, 1, bias=False)
        features = torch.ones(2, 1, 4, 4)
        target = torch.zeros(2, 1, 4, 4)
        rgb = torch.full((2, 3, 4, 4), 127, dtype=torch.uint8)
        config = {"loss": {"objective": "latent_mse_rgb_lpips", "latent_mse": 1,
                           "rgb_mse": 5, "perceptual": 2}}
        loss, parts = objective(adapter(features), target, rgb, Decoder(), Perceptual(), config)
        expected = parts["latent_mse"] + 5 * parts["rgb_mse"] + 2 * parts["rgb_lpips"]
        self.assertTrue(torch.allclose(loss, expected))
        self.assertTrue(torch.allclose(parts["weighted_rgb_lpips"], 2 * parts["rgb_lpips"]))
        loss.backward()
        self.assertGreater(float(adapter.weight.grad.abs().sum()), 0)

    def test_jepawms_style_zero_latent_weight_excludes_latent_supervision(self):
        class Decoder:
            def decode(self, latent):
                return latent.repeat(1, 3, 1, 1)

        class Perceptual(torch.nn.Module):
            def forward(self, prediction, target):
                return (prediction - target).abs().mean(dim=(1, 2, 3), keepdim=True)

        prediction = torch.full((1, 1, 4, 4), 0.25, requires_grad=True)
        rgb = torch.full((1, 3, 4, 4), 127, dtype=torch.uint8)
        config = {"loss": {"objective": "latent_mse_rgb_lpips", "latent_mse": 0,
                           "rgb_mse": 10, "perceptual": 1}}
        first, parts = objective(prediction, torch.zeros_like(prediction), rgb,
                                 Decoder(), Perceptual(), config)
        second, _ = objective(prediction, torch.ones_like(prediction) * 100, rgb,
                              Decoder(), Perceptual(), config)
        self.assertTrue(torch.allclose(first, second))
        self.assertTrue(torch.allclose(first, 10 * parts["rgb_mse"] + parts["rgb_lpips"]))
        first.backward()
        self.assertGreater(float(prediction.grad.abs().sum()), 0)

    def test_75k_jepawms_configs_reuse_mse_split_and_source(self):
        base = load_config("configs/experiments/vjepa21_cosmos_predicted_1frame_75k_drive_walk_mse.yaml")
        configs = [load_config(path) for path in (
            "configs/experiments/vjepa21_cosmos_predicted_1frame_75k_jepawms_rgb_lpips.yaml",
            "configs/experiments/vjepa21_cosmos_predicted_1frame_75k_jepawms_rgb_lpips_latent_mse.yaml",
        )]
        for config in configs:
            self.assertEqual(config["data"], base["data"])
            self.assertEqual(config["vjepa21"], base["vjepa21"])
            self.assertEqual(config["cosmos"], base["cosmos"])
            self.assertEqual(config["adapter"], base["adapter"])
            self.assertEqual(config["training"]["epochs"], 3)
            self.assertEqual(config["loss"]["perceptual_backbone"], "vgg")
            self.assertEqual(config["loss"]["evaluation_perceptual_backbone"], "alex")
            self.assertEqual(config["training"]["init_adapter_repo_id"], base["huggingface"]["repo_id"])
        self.assertEqual(configs[0]["loss"]["latent_mse"], 0)
        self.assertGreater(configs[1]["loss"]["latent_mse"], 0)
        self.assertEqual(len({c["experiment"]["output_dir"] for c in configs}), 2)
        self.assertEqual(len({c["tracking"]["run_name"] for c in configs}), 2)
        self.assertEqual(len({c["huggingface"]["repo_id"] for c in configs}), 2)

    def test_75k_high_lr_mse_continuation_is_isolated_and_compatible(self):
        base = load_config("configs/experiments/vjepa21_cosmos_predicted_1frame_75k_drive_walk_mse.yaml")
        continuation = load_config(
            "configs/experiments/vjepa21_cosmos_predicted_1frame_75k_drive_walk_mse_high_lr_continue.yaml"
        )
        for section in ("data", "vjepa21", "cosmos", "adapter"):
            self.assertEqual(continuation[section], base[section])
        self.assertEqual(continuation["training"]["init_adapter_repo_id"], base["huggingface"]["repo_id"])
        self.assertEqual(continuation["loss"]["objective"], "latent_mse")
        self.assertEqual(continuation["training"]["selection_metric"], "predicted_latent_mse")
        self.assertEqual(continuation["training"]["epochs"], 5)
        self.assertEqual(continuation["training"]["learning_rate"], 2e-4)
        self.assertNotEqual(continuation["experiment"]["output_dir"], base["experiment"]["output_dir"])
        self.assertNotEqual(continuation["tracking"]["run_name"], base["tracking"]["run_name"])
        self.assertNotEqual(continuation["huggingface"]["repo_id"], base["huggingface"]["repo_id"])

    def test_warm_start_loads_only_compatible_adapter_weights(self):
        config = {
            "data": {"cache_root": "cache", "train_samples": 2, "val_samples": 1,
                     "test_samples": 1, "context_frames": 14, "target_frame_offset": 0,
                     "crop_size": 384},
            "vjepa21": {"model_id": "test", "embed_dim": 1, "patch_size": 16,
                        "tubelet_size": 2},
            "cosmos": {"model_id": "test", "latent_channels": 1,
                       "spatial_compression": 8},
            "adapter": {"input_dim": 1, "hidden_dim": 1, "output_channels": 1,
                        "residual_blocks": 1},
            "training": {"input_latent": "predicted"},
            "loss": {"objective": "latent_mse_rgb_lpips"},
        }
        source_config = deepcopy(config)
        source_config["loss"]["objective"] = "latent_mse"
        source_adapter = torch.nn.Conv2d(1, 1, 1)
        new_adapter = torch.nn.Conv2d(1, 1, 1)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best.pt"
            torch.save({"adapter": source_adapter.state_dict(), "config": source_config,
                        "epoch": 4, "optimizer_step": 18000,
                        "optimizer": {"do_not_restore": True}}, checkpoint)
            report = load_adapter_initialization(new_adapter, config, checkpoint)
            self.assertEqual(report["source_optimizer_step"], 18000)
            for name, value in source_adapter.state_dict().items():
                self.assertTrue(torch.equal(value, new_adapter.state_dict()[name]))
            wrong_config = deepcopy(config)
            wrong_config["data"]["cache_root"] = "different-cache"
            with self.assertRaisesRegex(ValueError, "data.cache_root"):
                load_adapter_initialization(new_adapter, wrong_config, checkpoint)


if __name__ == "__main__":
    unittest.main()
