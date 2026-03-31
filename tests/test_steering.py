"""Tests for feature_search/steering.py — uses mock model/SAE, no real model required."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from feature_search.steering import get_steering_vector, steering_hook


# ---------------------------------------------------------------------------
# Minimal mock Gemma-like model for testing the hook
# ---------------------------------------------------------------------------

class _MockLayer(nn.Module):
    """Simulates a transformer decoder layer: adds a learned bias."""
    def __init__(self, d_model: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x, **kwargs):
        # Return a tuple like real Gemma layers do
        return (x + self.bias, None)


class _MockModel(nn.Module):
    def __init__(self, d_model: int = 64, n_layers: int = 4):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_MockLayer(d_model) for _ in range(n_layers)])

    @property
    def device(self):
        return next(self.parameters()).device


# ---------------------------------------------------------------------------
# get_steering_vector
# ---------------------------------------------------------------------------

class TestGetSteeringVector:
    def test_shape(self, mock_sae):
        sv = get_steering_vector(mock_sae, latent_idx=0)
        d_model = mock_sae.W_dec.shape[1]
        assert sv.shape == (d_model,)

    def test_matches_w_dec_row(self, mock_sae):
        idx = 7
        sv = get_steering_vector(mock_sae, latent_idx=idx)
        torch.testing.assert_close(sv, mock_sae.W_dec[idx].detach())

    def test_different_indices_different_vectors(self, mock_sae):
        sv0 = get_steering_vector(mock_sae, latent_idx=0)
        sv1 = get_steering_vector(mock_sae, latent_idx=1)
        assert not torch.allclose(sv0, sv1)

    def test_returns_detached_tensor(self, mock_sae):
        sv = get_steering_vector(mock_sae, latent_idx=3)
        assert not sv.requires_grad


# ---------------------------------------------------------------------------
# steering_hook
# ---------------------------------------------------------------------------

class TestSteeringHook:
    def _run_with_hook(self, model, layer_idx, sv, alpha, x):
        """Run model.model.layers[layer_idx] directly with hook active."""
        with steering_hook(model, layer_idx, sv, alpha):
            out, _ = model.model.layers[layer_idx](x)
        return out

    def test_hook_adds_scaled_vector(self):
        d_model = 16
        model = _MockModel(d_model=d_model, n_layers=3)
        x = torch.zeros(1, 5, d_model)  # batch=1, seq=5
        sv = torch.ones(d_model)
        alpha = 3.0

        out = self._run_with_hook(model, layer_idx=1, sv=sv, alpha=alpha, x=x)
        expected = x + alpha  # sv=ones, alpha=3 → adds 3.0 to each element
        torch.testing.assert_close(out, expected)

    def test_hook_removed_after_context(self):
        d_model = 16
        model = _MockModel(d_model=d_model, n_layers=3)
        x = torch.zeros(1, 4, d_model)
        sv = torch.ones(d_model) * 10.0

        with steering_hook(model, target_layer=0, steering_vector=sv, alpha=1.0):
            pass  # context exits immediately

        # After context, no hook — output should be just the layer bias (zero)
        out, _ = model.model.layers[0](x)
        torch.testing.assert_close(out, x)

    def test_alpha_zero_is_noop(self):
        d_model = 16
        model = _MockModel(d_model=d_model, n_layers=2)
        x = torch.randn(1, 6, d_model)
        sv = torch.randn(d_model)

        out_hooked = self._run_with_hook(model, 0, sv, alpha=0.0, x=x)
        out_baseline, _ = model.model.layers[0](x)
        torch.testing.assert_close(out_hooked, out_baseline)

    def test_negative_alpha_subtracts(self):
        d_model = 8
        model = _MockModel(d_model=d_model, n_layers=2)
        x = torch.ones(1, 3, d_model) * 5.0
        sv = torch.ones(d_model)

        out_pos = self._run_with_hook(model, 0, sv, alpha=+1.0, x=x)
        out_neg = self._run_with_hook(model, 0, sv, alpha=-1.0, x=x)
        # pos should be larger by 2.0 per element
        torch.testing.assert_close(out_pos - out_neg, torch.ones_like(out_pos) * 2.0)

    def test_hook_resolves_device_from_tensor(self):
        """Device resolution happens inside hook_fn — verify no crash when sv starts on CPU
        but hidden state could be on a different device (here both CPU, just verifying API)."""
        d_model = 16
        model = _MockModel(d_model=d_model, n_layers=2)
        x = torch.zeros(1, 4, d_model)
        sv = torch.ones(d_model)  # on CPU

        # Should not raise even though we don't call .to(model.device) before the hook
        out = self._run_with_hook(model, 0, sv, alpha=1.0, x=x)
        assert out.shape == x.shape

    def test_hook_handles_non_tuple_output(self):
        """Verify the non-tuple branch of hook_fn works."""
        d_model = 8

        class _TupleFreeLayers(nn.Module):
            def forward(self, x, **kwargs):
                return x  # returns tensor, not tuple

        model = _MockModel(d_model=d_model, n_layers=1)
        model.model.layers[0] = _TupleFreeLayers()
        x = torch.zeros(1, 3, d_model)
        sv = torch.ones(d_model)
        alpha = 2.0

        with steering_hook(model, target_layer=0, steering_vector=sv, alpha=alpha):
            out = model.model.layers[0](x)

        torch.testing.assert_close(out, x + alpha)
