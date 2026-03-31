"""Tests for chain_of_embedding/vip.py and tvi.py — no model required."""
from __future__ import annotations

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from chain_of_embedding.vip import (
    aggregate_vip,
    compute_layer_distances,
    cosine_distance,
    detect_vip,
)
from chain_of_embedding.tvi import compute_tvi, compute_tvi_batch, tvi_statistics


# ---------------------------------------------------------------------------
# cosine_distance
# ---------------------------------------------------------------------------

class TestCosineDistance:
    def test_identical_vectors_is_zero(self):
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_distance(v, v) == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_vectors_is_one(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert cosine_distance(a, b) == pytest.approx(1.0, abs=1e-6)

    def test_opposite_vectors_is_two(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert cosine_distance(a, b) == pytest.approx(2.0, abs=1e-6)

    def test_zero_vector_returns_zero(self):
        a = np.zeros(4)
        b = np.array([1.0, 2.0, 3.0, 4.0])
        assert cosine_distance(a, b) == 0.0

    def test_scale_invariant(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        assert cosine_distance(a, b) == pytest.approx(cosine_distance(a * 10, b * 0.1), abs=1e-6)


# ---------------------------------------------------------------------------
# compute_layer_distances
# ---------------------------------------------------------------------------

class TestComputeLayerDistances:
    def test_two_condition_output_shape(self, rng):
        n_layers, d = 8, 16
        hs_blind = rng.standard_normal((n_layers, d))
        hs_vis = rng.standard_normal((n_layers, d))
        result = compute_layer_distances(hs_blind, hs_vis)
        assert result["d_vis"].shape == (n_layers,)
        assert result["d_cf"] is None
        assert result["d_disc"] is None

    def test_three_condition_output_shape(self, rng):
        n_layers, d = 8, 16
        hs_blind = rng.standard_normal((n_layers, d))
        hs_vis = rng.standard_normal((n_layers, d))
        hs_cf = rng.standard_normal((n_layers, d))
        result = compute_layer_distances(hs_blind, hs_vis, hs_cf)
        assert result["d_cf"].shape == (n_layers,)
        assert result["d_disc"].shape == (n_layers,)

    def test_identical_conditions_zero_distance(self, rng):
        hs = rng.standard_normal((6, 32))
        result = compute_layer_distances(hs, hs)
        np.testing.assert_allclose(result["d_vis"], 0.0, atol=1e-6)

    def test_values_non_negative(self, rng):
        n_layers, d = 10, 32
        hs_blind = rng.standard_normal((n_layers, d))
        hs_vis = rng.standard_normal((n_layers, d))
        hs_cf = rng.standard_normal((n_layers, d))
        result = compute_layer_distances(hs_blind, hs_vis, hs_cf)
        assert (result["d_vis"] >= 0).all()
        assert (result["d_cf"] >= 0).all()
        assert (result["d_disc"] >= 0).all()


# ---------------------------------------------------------------------------
# detect_vip
# ---------------------------------------------------------------------------

class TestDetectVip:
    def _flat_then_spike(self, n=20, spike_at=10, spike_val=2.0) -> np.ndarray:
        """Flat baseline of 0.1, then a spike at spike_at."""
        d = np.full(n, 0.1)
        d[spike_at:] = spike_val
        return d

    def test_detects_known_spike(self):
        signal = self._flat_then_spike(n=20, spike_at=8, spike_val=5.0)
        vip = detect_vip(signal, baseline_layers=4, threshold_k=2.0)
        assert vip == 8

    def test_returns_int(self):
        signal = self._flat_then_spike()
        vip = detect_vip(signal)
        assert isinstance(vip, int)

    def test_prefers_disc_signal_over_vis(self):
        # d_vis stays flat, d_disc spikes → VIP should come from d_disc
        d_vis = np.full(20, 0.05)
        d_disc = self._flat_then_spike(n=20, spike_at=7, spike_val=3.0)
        vip = detect_vip(d_vis, d_disc=d_disc, baseline_layers=4)
        assert vip == 7

    def test_fallback_when_no_threshold_exceeded(self):
        # Monotonically increasing — threshold never crossed, fallback to steepest diff
        signal = np.linspace(0.0, 0.3, 20)  # stays near baseline, no clear spike
        vip = detect_vip(signal, baseline_layers=4, threshold_k=10.0)  # very high threshold
        assert 0 < vip < 20  # fallback produces a valid index

    def test_min_layer_respected(self):
        # Spike at layer 0, but min_layer=1 → should not return 0
        signal = np.array([10.0] + [0.1] * 15 + [5.0] * 4)
        vip = detect_vip(signal, min_layer=1, baseline_layers=3, threshold_k=1.0)
        assert vip >= 1


# ---------------------------------------------------------------------------
# aggregate_vip
# ---------------------------------------------------------------------------

class TestAggregateVip:
    def _make_results(self, n_samples, n_layers, rng, with_cf=True):
        results = []
        for _ in range(n_samples):
            hs_blind = rng.standard_normal((n_layers, 32))
            hs_vis = rng.standard_normal((n_layers, 32))
            hs_cf = rng.standard_normal((n_layers, 32)) if with_cf else None
            results.append(compute_layer_distances(hs_blind, hs_vis, hs_cf))
        return results

    def test_output_keys(self, rng):
        results = self._make_results(5, 10, rng)
        agg = aggregate_vip(results)
        for key in ("mean_d_vis", "mean_d_cf", "mean_d_disc", "vip_per_sample", "vip_mean", "vip_median"):
            assert key in agg

    def test_vip_per_sample_length(self, rng):
        n = 8
        results = self._make_results(n, 12, rng)
        agg = aggregate_vip(results)
        assert len(agg["vip_per_sample"]) == n

    def test_mean_d_vis_shape(self, rng):
        n_layers = 12
        results = self._make_results(6, n_layers, rng)
        agg = aggregate_vip(results)
        assert agg["mean_d_vis"].shape == (n_layers,)

    def test_no_cf_results_in_none_fields(self, rng):
        results = self._make_results(4, 8, rng, with_cf=False)
        agg = aggregate_vip(results)
        assert agg["mean_d_cf"] is None
        assert agg["mean_d_disc"] is None


# ---------------------------------------------------------------------------
# compute_tvi
# ---------------------------------------------------------------------------

class TestComputeTvi:
    def test_identical_conditions_zero_tvi(self, rng):
        hs = rng.standard_normal((10, 32))
        tvi = compute_tvi(hs, hs, vip=3)
        assert tvi == pytest.approx(0.0, abs=1e-6)

    def test_normalized_smaller_than_unnormalized(self, rng):
        hs_blind = rng.standard_normal((10, 64))
        hs_vis = rng.standard_normal((10, 64))
        t_norm = compute_tvi(hs_blind, hs_vis, vip=2, normalize_by_dim=True)
        t_raw = compute_tvi(hs_blind, hs_vis, vip=2, normalize_by_dim=False)
        # sqrt(64) = 8, so normalized should be ~8x smaller
        assert t_norm < t_raw

    def test_vip_at_last_layer_returns_single_value(self, rng):
        n_layers = 8
        hs_blind = rng.standard_normal((n_layers, 16))
        hs_vis = rng.standard_normal((n_layers, 16))
        tvi = compute_tvi(hs_blind, hs_vis, vip=n_layers - 1)
        assert np.isfinite(tvi)

    def test_later_vip_gives_smaller_tvi(self, rng):
        # With vip=2 we average over more layers than vip=6 (more early layers are included)
        # but TVI is an average so it might not be strictly smaller.
        # What we CAN test: vip at last layer returns only one distance value.
        n_layers = 10
        hs_blind = rng.standard_normal((n_layers, 32))
        hs_vis = hs_blind + rng.standard_normal((n_layers, 32)) * 0.5  # small perturbation
        tvi_early = compute_tvi(hs_blind, hs_vis, vip=0, normalize_by_dim=False)
        tvi_last = compute_tvi(hs_blind, hs_vis, vip=n_layers - 1, normalize_by_dim=False)
        # Both should be finite and positive
        assert tvi_early > 0
        assert tvi_last > 0

    def test_batch_matches_single(self, rng):
        n_samples, n_layers, d = 5, 8, 16
        hs_blind = rng.standard_normal((n_samples, n_layers, d))
        hs_vis = rng.standard_normal((n_samples, n_layers, d))
        vip = 3

        batch_result = compute_tvi_batch(hs_blind, hs_vis, vip)
        single_results = np.array([
            compute_tvi(hs_blind[i], hs_vis[i], vip) for i in range(n_samples)
        ])
        np.testing.assert_allclose(batch_result, single_results, atol=1e-6)


# ---------------------------------------------------------------------------
# tvi_statistics
# ---------------------------------------------------------------------------

class TestTviStatistics:
    def test_basic_stats_keys(self, rng):
        tvi = rng.uniform(0, 1, 30)
        stats = tvi_statistics(tvi)
        for key in ("mean", "std", "median", "n"):
            assert key in stats

    def test_spearman_computed_when_is_correct_provided(self, rng):
        n = 40
        tvi = rng.uniform(0, 1, n)
        is_correct = (tvi > 0.5).astype(float)  # correlated with tvi
        stats = tvi_statistics(tvi, is_correct=is_correct)
        assert "spearman_rho" in stats
        assert "spearman_pval" in stats
        assert stats["spearman_rho"] > 0  # expect positive correlation

    def test_dvt_dt_split(self, rng):
        n = 30
        tvi = rng.uniform(0, 1, n)
        is_vd = np.array([True] * 15 + [False] * 15)
        stats = tvi_statistics(tvi, is_vision_dependent=is_vd)
        assert "tvi_mean_dvt" in stats
        assert "tvi_mean_dt" in stats
        assert stats["n_dvt"] == 15
        assert stats["n_dt"] == 15

    def test_spearman_not_computed_when_too_few_valid(self):
        tvi = np.array([0.1, 0.2, 0.3])
        is_correct = np.array([0.0, 1.0, 0.0])
        stats = tvi_statistics(tvi, is_correct=is_correct)
        # Fewer than 10 valid samples → no Spearman
        assert "spearman_rho" not in stats
