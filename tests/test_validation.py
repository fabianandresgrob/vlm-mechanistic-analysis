"""Tests for feature_search/validation.py — no model required."""
from __future__ import annotations

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from feature_search.validation import (
    feature_activation_test,
    rank_biserial_correlation,
    test_condition_divergence,
)


class TestRankBiserialCorrelation:
    def test_known_value(self):
        # r = 2*U1/(n1*n2) - 1
        # U1 = n1*n2 (all x > y) → r = +1
        r = rank_biserial_correlation(u_stat=100, n1=10, n2=10)
        assert r == pytest.approx(1.0, abs=1e-6)

    def test_u_equals_half_max_gives_zero(self):
        # U1 = n1*n2/2 → r = 0
        r = rank_biserial_correlation(u_stat=50, n1=10, n2=10)
        assert r == pytest.approx(0.0, abs=1e-6)

    def test_u_equals_zero_gives_minus_one(self):
        # U1 = 0 (all y > x) → r = -1
        r = rank_biserial_correlation(u_stat=0, n1=10, n2=10)
        assert r == pytest.approx(-1.0, abs=1e-6)

    def test_range(self, rng):
        for _ in range(20):
            n1, n2 = rng.integers(5, 50, size=2)
            u = rng.uniform(0, n1 * n2)
            r = rank_biserial_correlation(u, int(n1), int(n2))
            assert -1.0 - 1e-6 <= r <= 1.0 + 1e-6


class TestConditionDivergence:
    def test_separated_groups_low_pvalue(self, rng):
        # Clearly separated: correct >> biased
        correct = rng.uniform(0.8, 1.0, 30)
        biased = rng.uniform(0.0, 0.2, 30)
        result = test_condition_divergence(correct, biased, alternative="greater")
        assert result["pvalue"] < 0.001
        assert result["effect_size_r"] > 0.5

    def test_identical_groups_high_pvalue(self, rng):
        scores = rng.uniform(0, 1, 40)
        result = test_condition_divergence(scores[:20], scores[20:], alternative="greater")
        assert result["pvalue"] > 0.05

    def test_output_keys(self, rng):
        a = rng.uniform(0, 1, 20)
        b = rng.uniform(0, 1, 20)
        result = test_condition_divergence(a, b)
        for key in ("u_stat", "pvalue", "effect_size_r", "n_correct", "n_biased", "mean_correct", "mean_biased"):
            assert key in result

    def test_n_values_match_inputs(self, rng):
        correct = rng.uniform(0, 1, 15)
        biased = rng.uniform(0, 1, 25)
        result = test_condition_divergence(correct, biased)
        assert result["n_correct"] == 15
        assert result["n_biased"] == 25

    def test_mean_values_correct(self, rng):
        correct = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        biased = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
        result = test_condition_divergence(correct, biased)
        assert result["mean_correct"] == pytest.approx(np.mean(correct))
        assert result["mean_biased"] == pytest.approx(np.mean(biased))


class TestFeatureActivationTest:
    def _make_feat_acts(self, rng, n_samples, d_sae, correct_mask):
        """Create activations where 'correct' samples have higher activation for latents 0-4."""
        acts = rng.uniform(0.0, 0.1, (n_samples, d_sae)).astype(np.float32)
        acts[correct_mask, :5] += 0.8   # strong signal for first 5 latents
        return acts

    def test_returns_list_of_dicts(self, rng):
        n, d = 40, 50
        is_correct = (rng.uniform(size=n) > 0.5).astype(float)
        correct_mask = is_correct.astype(bool)
        acts = self._make_feat_acts(rng, n, d, correct_mask)
        results = feature_activation_test(acts, None, latent_indices=[0, 1, 2], is_correct=is_correct)
        assert isinstance(results, list)
        assert len(results) == 3
        assert all("latent_idx" in r for r in results)

    def test_sorted_by_effect_size_descending(self, rng):
        n, d = 60, 50
        is_correct = (rng.uniform(size=n) > 0.5).astype(float)
        acts = rng.uniform(0, 1, (n, d)).astype(np.float32)
        results = feature_activation_test(acts, None, latent_indices=list(range(10)), is_correct=is_correct)
        effect_sizes = [r["effect_size_r"] for r in results]
        assert effect_sizes == sorted(effect_sizes, reverse=True)

    def test_strongly_separated_features_rank_first(self, rng):
        n, d = 80, 50
        is_correct = np.array([1.0] * 40 + [0.0] * 40)
        correct_mask = is_correct.astype(bool)
        acts = self._make_feat_acts(rng, n, d, correct_mask)
        results = feature_activation_test(acts, None, latent_indices=list(range(10)), is_correct=is_correct)
        # The strongly-separated latents (0-4) should rank higher than random ones (5-9)
        top_latents = {r["latent_idx"] for r in results[:5]}
        assert len(top_latents & {0, 1, 2, 3, 4}) >= 3

    def test_latent_idx_preserved(self, rng):
        n, d = 30, 20
        is_correct = (rng.uniform(size=n) > 0.5).astype(float)
        acts = rng.uniform(0, 1, (n, d)).astype(np.float32)
        latents = [3, 7, 12]
        results = feature_activation_test(acts, None, latent_indices=latents, is_correct=is_correct)
        returned_latents = {r["latent_idx"] for r in results}
        assert returned_latents == set(latents)
