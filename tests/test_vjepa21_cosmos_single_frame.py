import json

import pytest
import torch

from experiments.vjepa21_cosmos_single_frame.adapter import VJEPA21ToCosmosImageAdapter
from experiments.vjepa21_cosmos_single_frame.data import CACHE_SCHEMA, LatentShardDataset
from experiments.vjepa21_cosmos_single_frame.models import validate_model_geometry


def test_single_frame_adapter_geometry_and_gradients():
    adapter = VJEPA21ToCosmosImageAdapter(32, 16, 8, residual_blocks=2, dropout=0.0)
    source = torch.randn(2, 32, 1, 6, 6, requires_grad=True)
    output = adapter(source, (12, 12))
    assert output.shape == (2, 8, 12, 12)
    output.mean().backward()
    assert source.grad is not None


def test_single_frame_adapter_rejects_two_slots():
    adapter = VJEPA21ToCosmosImageAdapter(32, 16, 8, residual_blocks=1, dropout=0.0)
    with pytest.raises(ValueError, match="T=1"):
        adapter(torch.randn(1, 32, 2, 6, 6), (12, 12))


def test_single_frame_geometry_contract():
    config = {
        "data": {"num_frames": 16, "context_frames": 14, "target_frame_offset": 0, "crop_size": 384},
        "vjepa21": {"tubelet_size": 2, "patch_size": 16, "embed_dim": 1408, "use_final_layer_only": True},
        "cosmos": {"spatial_compression": 8, "latent_channels": 16},
        "adapter": {"input_dim": 1408, "output_channels": 16},
    }
    validate_model_geometry(config)


def test_single_frame_cache_dataset(tmp_path):
    payload = {
        "keys": ["a"],
        "jepa_predicted": torch.randn(1, 32, 1, 6, 6),
        "jepa_target": torch.randn(1, 32, 1, 6, 6),
        "cosmos_target": torch.randn(1, 8, 12, 12),
        "target_rgb": torch.randint(0, 256, (1, 3, 96, 96), dtype=torch.uint8),
    }
    torch.save(payload, tmp_path / "latents-00000.pt")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "samples": 1,
                "metadata": {"schema": CACHE_SCHEMA},
                "shards": [{"file": "latents-00000.pt", "samples": 1, "keys": ["a"]}],
            }
        )
    )
    sample = next(iter(LatentShardDataset(tmp_path, shuffle=False, seed=1)))
    assert sample["jepa_predicted"].shape == (32, 1, 6, 6)
    assert sample["cosmos_target"].shape == (8, 12, 12)
    assert sample["target_rgb"].shape == (3, 96, 96)
