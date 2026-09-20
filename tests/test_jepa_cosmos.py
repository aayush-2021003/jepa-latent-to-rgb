import json

import torch

from experiments.jepa_cosmos.adapter import JEPAToCosmosAdapter
from experiments.jepa_cosmos.common import tokens_from_volume, volume_from_tokens
from experiments.jepa_cosmos.data import LatentShardDataset
from experiments.jepa_cosmos.losses import latent_alignment_loss


def test_token_volume_roundtrip():
    tokens = torch.randn(2, 4 * 3 * 5, 7)
    volume = volume_from_tokens(tokens, 4, 3, 5)
    assert volume.shape == (2, 7, 4, 3, 5)
    torch.testing.assert_close(tokens_from_volume(volume), tokens)


def test_adapter_target_geometry_and_gradients():
    adapter = JEPAToCosmosAdapter(32, 16, 8, residual_blocks=2, dropout=0.0)
    source = torch.randn(2, 32, 4, 6, 6, requires_grad=True)
    output = adapter(source, (2, 12, 12))
    assert output.shape == (2, 8, 2, 12, 12)
    output.mean().backward()
    assert source.grad is not None


def test_zero_latent_alignment_loss():
    target = torch.randn(2, 16, 3, 8, 8)
    loss, parts = latent_alignment_loss(target, target, 1.0, 0.1)
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-6, rtol=0)
    torch.testing.assert_close(parts["latent_l1"], torch.zeros_like(loss))


def test_latent_shard_dataset(tmp_path):
    payload = {
        "keys": ["a", "b"],
        "jepa_target": torch.randn(2, 4, 2, 3, 3),
        "jepa_predicted": torch.randn(2, 4, 2, 3, 3),
        "cosmos_anchor": torch.randn(2, 2, 1, 4, 4),
        "cosmos_target": torch.randn(2, 2, 2, 4, 4),
    }
    torch.save(payload, tmp_path / "latents-00000.pt")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "samples": 2,
                "metadata": {
                    "cosmos_target_schema": "last_context_anchor_plus_future_v1"
                },
                "shards": [{"file": "latents-00000.pt", "samples": 2, "keys": ["a", "b"]}],
            }
        )
    )
    dataset = LatentShardDataset(tmp_path, shuffle=False, seed=1)
    samples = list(dataset)
    assert [sample["key"] for sample in samples] == ["a", "b"]
    assert "jepa_predicted" in samples[0]
    assert samples[0]["cosmos_anchor"].shape[1] == 1
    assert samples[0]["cosmos_target"].shape[1] == 2

    limited = LatentShardDataset(tmp_path, shuffle=False, seed=1, max_samples=1)
    assert len(limited) == 1
    assert [sample["key"] for sample in limited] == ["a"]
