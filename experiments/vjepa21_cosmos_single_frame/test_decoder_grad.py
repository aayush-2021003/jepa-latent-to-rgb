"""Verify that frozen Cosmos decoding preserves adapter input gradients."""
from __future__ import annotations

import unittest

import torch

from experiments.vjepa21_cosmos_single_frame.models import CosmosContinuousImageTokenizer


class _InferenceOnlyTokenizer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._dec_model = torch.nn.Conv2d(2, 3, kernel_size=1, bias=False)
        self._dec_model.requires_grad_(False)

    @torch.no_grad()
    def decode(self, latent):
        return self._dec_model(latent)


class DecoderGradientTest(unittest.TestCase):
    def test_training_decode_reaches_input_but_not_frozen_weights(self):
        wrapper = CosmosContinuousImageTokenizer.__new__(CosmosContinuousImageTokenizer)
        wrapper.device = torch.device("cpu")
        wrapper.dtype = torch.float32
        wrapper.model = _InferenceOnlyTokenizer()
        latent = torch.randn(1, 2, 4, 4, requires_grad=True)
        image = wrapper.decode(latent)
        self.assertTrue(image.requires_grad)
        image.square().mean().backward()
        self.assertIsNotNone(latent.grad)
        self.assertGreater(float(latent.grad.abs().sum()), 0)
        self.assertIsNone(wrapper.model._dec_model.weight.grad)

    def test_inference_path_remains_non_differentiable(self):
        wrapper = CosmosContinuousImageTokenizer.__new__(CosmosContinuousImageTokenizer)
        wrapper.device = torch.device("cpu")
        wrapper.dtype = torch.float32
        wrapper.model = _InferenceOnlyTokenizer()
        with torch.no_grad():
            image = wrapper.decode(torch.randn(1, 2, 4, 4))
        self.assertFalse(image.requires_grad)


if __name__ == "__main__":
    unittest.main()
