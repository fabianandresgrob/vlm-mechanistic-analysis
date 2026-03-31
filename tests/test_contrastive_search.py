"""Tests for feature_search/contrastive_search.py — uses MockSAE, no model required."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from feature_search.contrastive_search import compute_activation_frequencies, separation_scores


class TestComputeActivationFrequencies:
    def test_output_shape(self, mock_sae, rng):
        acts = rng.standard_normal((20, 64)).astype(np.float32)
        freq = compute_activation_frequencies(acts, mock_sae, batch_size=8)
        assert freq.shape == (mock_sae.cfg.d_sae,)

    def test_frequencies_in_zero_one(self, mock_sae, rng):
        acts = rng.standard_normal((50, 64)).astype(np.float32)
        freq = compute_activation_frequencies(acts, mock_sae, batch_size=16)
        assert (freq >= 0).all() and (freq <= 1.0 + 1e-6).all()

    def test_zero_input_gives_zero_frequency(self, mock_sae):
        # All-zero input → encode produces zero activations → frequency 0
        acts = np.zeros((10, 64), dtype=np.float32)
        freq = compute_activation_frequencies(acts, mock_sae)
        np.testing.assert_allclose(freq, 0.0, atol=1e-6)

    def test_batch_boundary(self, mock_sae, rng):
        """Results should be the same regardless of batch_size."""
        acts = rng.standard_normal((17, 64)).astype(np.float32)
        freq_1 = compute_activation_frequencies(acts, mock_sae, batch_size=1)
        freq_8 = compute_activation_frequencies(acts, mock_sae, batch_size=8)
        freq_100 = compute_activation_frequencies(acts, mock_sae, batch_size=100)
        np.testing.assert_allclose(freq_1, freq_8, atol=1e-5)
        np.testing.assert_allclose(freq_1, freq_100, atol=1e-5)


class TestSeparationScores:
    def _acts(self, rng, n, d=64):
        return rng.standard_normal((n, d)).astype(np.float32)

    def test_output_keys(self, mock_sae, rng):
        result = separation_scores(self._acts(rng, 20), self._acts(rng, 20), mock_sae)
        for key in ("f_vis", "f_blind", "s_visual", "s_prior", "noise_mask", "top_visual", "top_prior"):
            assert key in result

    def test_s_visual_plus_s_prior_is_zero(self, mock_sae, rng):
        """s_visual + s_prior = (f_vis - f_blind) + (f_blind - f_vis) = 0."""
        result = separation_scores(self._acts(rng, 30), self._acts(rng, 30), mock_sae)
        np.testing.assert_allclose(result["s_visual"] + result["s_prior"], 0.0, atol=1e-6)

    def test_top_visual_length(self, mock_sae, rng):
        result = separation_scores(self._acts(rng, 20), self._acts(rng, 20), mock_sae)
        # top 100 requested, but d_sae=128 so we get min(100, 128)=100
        assert len(result["top_visual"]) == min(100, mock_sae.cfg.d_sae)
        assert len(result["top_prior"]) == min(100, mock_sae.cfg.d_sae)

    def test_noise_filter_reduces_candidates(self, mock_sae, rng):
        # Noise baseline with large acts → many latents will activate → filtered out
        noise = self._acts(rng, 200) * 5.0  # large values → high frequency
        result = separation_scores(
            self._acts(rng, 20), self._acts(rng, 20), mock_sae,
            noise_baseline_acts=noise, noise_threshold=0.0,  # threshold=0 → filter everything
        )
        # With threshold=0 and high-frequency noise, almost all should be masked out
        assert result["noise_mask"].sum() < mock_sae.cfg.d_sae

    def test_no_noise_filter_all_pass(self, mock_sae, rng):
        result = separation_scores(
            self._acts(rng, 20), self._acts(rng, 20), mock_sae,
            noise_baseline_acts=None,
        )
        assert result["noise_mask"].all()

    def test_top_visual_latents_have_highest_scores(self, mock_sae, rng):
        result = separation_scores(self._acts(rng, 30), self._acts(rng, 30), mock_sae)
        top_idx = result["top_visual"]
        top_scores = result["s_visual"][top_idx]
        # All top latents should have score >= any random other latent not in top
        min_top = top_scores.min()
        other_idx = np.setdiff1d(np.arange(mock_sae.cfg.d_sae), top_idx)
        if len(other_idx) > 0:
            assert min_top >= result["s_visual"][other_idx].max() - 1e-6
