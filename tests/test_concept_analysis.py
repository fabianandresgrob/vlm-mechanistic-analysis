"""Tests for sae_analysis/concept_analysis.py — pure computation, no model/data required."""
from __future__ import annotations

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sae_analysis.concept_analysis import (
    assign_top_k_concepts,
    build_vocabulary,
    compute_concept_frequencies,
    find_coverage_gaps,
    kl_divergence,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_samples_vab():
    return [
        {"topic": "counting", "sub_topic": "objects", "messages": [], "answer": "three"},
        {"topic": "geometry", "sub_topic": "shapes", "messages": [], "answer": "circle"},
        {"topic": "optical illusion", "sub_topic": "size", "messages": [], "answer": "big"},
    ]


def _make_samples_vlind():
    return [
        {
            "concept": "banana",
            "existent_noun": "banana",
            "non_existent_noun": "apple",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Is the banana ripe?"}]}],
            "answer": "true",
        },
        {
            "concept": "chair",
            "existent_noun": "chair",
            "non_existent_noun": "table",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Is there a chair here?"}]}],
            "answer": "false",
        },
    ]


def _make_samples_vilp():
    return [
        {
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Please answer with one word: How many dogs are in the image?"}]}],
            "answer": "three",
        },
        {
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Please answer with one word: What color is the car?"}]}],
            "answer": "red",
        },
    ]


# ─── build_vocabulary ─────────────────────────────────────────────────────────

class TestBuildVocabulary:
    def test_extracts_vab_fields(self):
        samples = {"vab": _make_samples_vab()}
        concepts, sources = build_vocabulary(samples)
        assert "counting" in concepts
        assert "geometry" in concepts
        assert "optical illusion" in concepts
        assert "objects" in concepts
        assert "shapes" in concepts

    def test_extracts_vlind_fields(self):
        samples = {"vlind_bench": _make_samples_vlind()}
        concepts, sources = build_vocabulary(samples)
        assert "banana" in concepts
        assert "chair" in concepts
        assert "apple" in concepts
        assert "table" in concepts

    def test_extracts_vilp_single_word_answers(self):
        samples = {"vilp": _make_samples_vilp()}
        concepts, sources = build_vocabulary(samples)
        # ViLP single-word answers ("three", "red") are the visual concepts being tested
        assert "three" in concepts
        assert "red" in concepts

    def test_includes_imagenet_classes(self):
        samples = {}
        classes = ["tench, Tinca tinca", "goldfish, Carassius auratus", "great white shark"]
        concepts, sources = build_vocabulary(samples, imagenet_classes=classes)
        assert "tench" in concepts
        assert "goldfish" in concepts
        assert "great white shark" in concepts

    def test_imagenet_class_source_tagged(self):
        samples = {}
        concepts, sources = build_vocabulary(samples, imagenet_classes=["tench, Tinca tinca"])
        assert "imagenet" in sources.get("tench", set())

    def test_extra_concepts(self):
        samples = {}
        concepts, _ = build_vocabulary(samples, extra_concepts=["optical illusion", "spatial reasoning"])
        assert "optical illusion" in concepts
        assert "spatial reasoning" in concepts

    def test_deduplication(self):
        # Same concept from two datasets should appear once
        samples = {
            "a": [{"concept": "banana", "messages": [], "answer": "x"}],
            "b": [{"concept": "banana", "messages": [], "answer": "x"}],
        }
        concepts, sources = build_vocabulary(samples)
        assert concepts.count("banana") == 1
        # But should be tagged with both sources
        assert sources["banana"] == {"a", "b"}

    def test_empty_input(self):
        concepts, sources = build_vocabulary({})
        assert isinstance(concepts, list)
        assert len(concepts) == 0

    def test_returns_sorted(self):
        samples = {"vab": _make_samples_vab()}
        concepts, _ = build_vocabulary(samples)
        assert concepts == sorted(concepts)

    def test_no_single_char_concepts(self):
        samples = {}
        concepts, _ = build_vocabulary(samples, extra_concepts=["a", "b", "ok", "cat"])
        assert "a" not in concepts
        assert "b" not in concepts


# ─── assign_top_k_concepts ────────────────────────────────────────────────────

class TestAssignTopKConcepts:
    def _make_embeddings(self, n: int, d: int = 16, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        emb = rng.standard_normal((n, d)).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=-1, keepdims=True)
        return emb

    def test_output_shape(self):
        imgs = self._make_embeddings(10)
        concepts = self._make_embeddings(50)
        result = assign_top_k_concepts(imgs, concepts, k=5)
        assert result.shape == (10, 5)

    def test_k_equals_one(self):
        imgs = self._make_embeddings(5)
        concepts = self._make_embeddings(20)
        result = assign_top_k_concepts(imgs, concepts, k=1)
        assert result.shape == (5, 1)

    def test_top_concept_is_most_similar(self):
        d = 16
        # Create image embedding that is exactly concept[3]
        rng = np.random.default_rng(42)
        concepts = rng.standard_normal((10, d)).astype(np.float32)
        concepts /= np.linalg.norm(concepts, axis=-1, keepdims=True)
        img = concepts[3:4].copy()  # identical to concept 3
        result = assign_top_k_concepts(img, concepts, k=1)
        assert result[0, 0] == 3

    def test_indices_in_valid_range(self):
        imgs = self._make_embeddings(8)
        n_concepts = 30
        concepts = self._make_embeddings(n_concepts)
        result = assign_top_k_concepts(imgs, concepts, k=5)
        assert result.min() >= 0
        assert result.max() < n_concepts

    def test_no_duplicate_concepts_per_image(self):
        imgs = self._make_embeddings(10)
        concepts = self._make_embeddings(50)
        result = assign_top_k_concepts(imgs, concepts, k=5)
        for row in result:
            assert len(set(row)) == len(row), "Duplicate concept in top-K"


# ─── compute_concept_frequencies ─────────────────────────────────────────────

class TestComputeConceptFrequencies:
    def test_sums_to_one_when_normalized(self):
        assignments = np.array([[0, 1, 2], [0, 3, 4]])
        freq = compute_concept_frequencies(assignments, n_concepts=10, normalize=True)
        assert abs(freq.sum() - 1.0) < 1e-6

    def test_raw_counts_when_not_normalized(self):
        assignments = np.array([[0, 0, 1]])
        freq = compute_concept_frequencies(assignments, n_concepts=5, normalize=False)
        assert freq[0] == 2
        assert freq[1] == 1
        assert freq[2:].sum() == 0

    def test_output_length(self):
        assignments = np.array([[0, 1], [2, 3]])
        freq = compute_concept_frequencies(assignments, n_concepts=100, normalize=True)
        assert len(freq) == 100

    def test_uniform_assignments(self):
        # Assign every concept once
        n_concepts = 10
        assignments = np.arange(n_concepts).reshape(1, -1)
        freq = compute_concept_frequencies(assignments, n_concepts=n_concepts, normalize=True)
        np.testing.assert_allclose(freq, np.full(n_concepts, 1.0 / n_concepts), atol=1e-6)


# ─── kl_divergence ────────────────────────────────────────────────────────────

class TestKLDivergence:
    def test_identical_distributions_near_zero(self):
        p = np.array([0.5, 0.3, 0.2])
        assert kl_divergence(p, p.copy()) < 1e-6

    def test_kl_is_non_negative(self):
        rng = np.random.default_rng(0)
        for _ in range(10):
            p = rng.dirichlet(np.ones(20))
            q = rng.dirichlet(np.ones(20))
            assert kl_divergence(p, q) >= 0

    def test_asymmetric(self):
        # Mirror-image pairs give equal KL — use truly non-symmetric distributions
        p = np.array([0.7, 0.2, 0.1])
        q = np.array([0.1, 0.6, 0.3])
        assert kl_divergence(p, q) != pytest.approx(kl_divergence(q, p), abs=0.01)


# ─── find_coverage_gaps ───────────────────────────────────────────────────────

class TestFindCoverageGaps:
    def _make_freqs(self, n: int = 20) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(0)
        bench = rng.dirichlet(np.ones(n))
        train = rng.dirichlet(np.ones(n))
        return bench, train

    def test_output_keys(self):
        n = 20
        bench, train = self._make_freqs(n)
        concepts = [f"c{i}" for i in range(n)]
        result = find_coverage_gaps(bench, train, concepts, top_n=5)
        assert "over_represented" in result
        assert "under_represented" in result
        assert "gap_scores" in result
        assert "kl_divergence" in result

    def test_over_represented_has_positive_gap(self):
        n = 20
        bench, train = self._make_freqs(n)
        concepts = [f"c{i}" for i in range(n)]
        result = find_coverage_gaps(bench, train, concepts, top_n=5)
        for item in result["over_represented"]:
            assert item["gap"] >= 0

    def test_under_represented_has_negative_gap(self):
        n = 20
        bench, train = self._make_freqs(n)
        concepts = [f"c{i}" for i in range(n)]
        result = find_coverage_gaps(bench, train, concepts, top_n=5)
        for item in result["under_represented"]:
            assert item["gap"] <= 0

    def test_top_n_respected(self):
        n = 50
        bench, train = self._make_freqs(n)
        concepts = [f"c{i}" for i in range(n)]
        result = find_coverage_gaps(bench, train, concepts, top_n=10)
        assert len(result["over_represented"]) == 10
        assert len(result["under_represented"]) == 10

    def test_gap_scores_length(self):
        n = 15
        bench, train = self._make_freqs(n)
        concepts = [f"c{i}" for i in range(n)]
        result = find_coverage_gaps(bench, train, concepts)
        assert len(result["gap_scores"]) == n

    def test_most_overrepresented_concept_identified(self):
        n = 10
        concepts = [f"c{i}" for i in range(n)]
        bench = np.zeros(n)
        train = np.zeros(n)
        bench[3] = 1.0  # concept 3 is entirely in benchmark
        train[7] = 1.0  # concept 7 is entirely in training
        result = find_coverage_gaps(bench, train, concepts, top_n=3)
        assert result["over_represented"][0]["concept"] == "c3"
