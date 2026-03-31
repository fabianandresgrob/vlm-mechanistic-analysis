"""Tests for eva/js_divergence.py — pure computation, no model required."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from eva.js_divergence import (
    correlate_with_correctness,
    find_peak_layer,
    jensen_shannon_divergence,
)


class TestJensenShannonDivergence:
    def _uniform(self, vocab: int) -> torch.Tensor:
        return torch.full((vocab,), 1.0 / vocab)

    def _onehot(self, idx: int, vocab: int) -> torch.Tensor:
        t = torch.zeros(vocab)
        t[idx] = 1.0
        return t

    def test_identical_distributions_is_zero(self):
        p = self._uniform(100)
        jsd = jensen_shannon_divergence(p, p.clone())
        assert jsd.item() == pytest.approx(0.0, abs=1e-5)

    def test_symmetric(self):
        p = self._uniform(100)
        q = F.softmax(torch.randn(100), dim=-1)
        assert jensen_shannon_divergence(p, q).item() == pytest.approx(
            jensen_shannon_divergence(q, p).item(), abs=1e-5
        )

    def test_bounded_between_zero_and_one(self):
        for _ in range(10):
            p = F.softmax(torch.randn(200), dim=-1)
            q = F.softmax(torch.randn(200), dim=-1)
            jsd = jensen_shannon_divergence(p, q).item()
            assert 0.0 <= jsd <= 1.0 + 1e-5

    def test_maximally_different_distributions(self):
        # Two dirac deltas on different tokens → JSD = 1 bit
        p = self._onehot(0, 100)
        q = self._onehot(99, 100)
        jsd = jensen_shannon_divergence(p, q).item()
        assert jsd == pytest.approx(1.0, abs=1e-4)

    def test_batched_input(self):
        p = F.softmax(torch.randn(4, 50), dim=-1)
        q = F.softmax(torch.randn(4, 50), dim=-1)
        jsd = jensen_shannon_divergence(p, q)
        assert jsd.shape == (4,)
        assert (jsd >= 0).all()


class TestFindPeakLayer:
    def test_returns_argmax(self):
        curve = np.array([0.1, 0.3, 0.8, 0.5, 0.2])
        assert find_peak_layer(curve) == 2

    def test_single_layer(self):
        assert find_peak_layer(np.array([0.7])) == 0

    def test_peak_at_last_layer(self):
        curve = np.array([0.1, 0.2, 0.3, 0.9])
        assert find_peak_layer(curve) == 3

    def test_all_equal_returns_zero(self):
        assert find_peak_layer(np.ones(5)) == 0


class TestCorrelateWithCorrectness:
    def test_perfect_positive_correlation(self):
        n = 50
        js_per_layer = np.zeros((n, 5))
        js_per_layer[:, 2] = np.linspace(0, 1, n)
        is_correct = np.linspace(0, 1, n)
        rho, pval = correlate_with_correctness(js_per_layer, is_correct, layer_idx=2)
        assert rho == pytest.approx(1.0, abs=0.01)
        assert pval < 0.001

    def test_nan_correctness_labels_are_dropped(self):
        n = 30
        js_per_layer = np.tile(np.linspace(0, 1, n)[:, None], (1, 3))
        is_correct = np.array([float("nan")] * 10 + list(np.linspace(0, 1, 20)))
        rho, pval = correlate_with_correctness(js_per_layer, is_correct, layer_idx=0)
        assert not np.isnan(rho)

    def test_too_few_samples_returns_nan(self):
        js_per_layer = np.ones((5, 3))
        is_correct = np.ones(5)
        rho, pval = correlate_with_correctness(js_per_layer, is_correct, layer_idx=0)
        assert np.isnan(rho)
