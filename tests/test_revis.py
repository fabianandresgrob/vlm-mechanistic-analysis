"""Tests for revis/vector_calculator.py — pure computation, no model required."""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch

from revis.vector_calculator import gram_schmidt_orthogonalize, compute_revis_vector


class TestGramSchmidtOrthogonalize:
    def test_result_is_orthogonal_to_basis(self):
        torch.manual_seed(0)
        v = torch.randn(64)
        basis = torch.randn(64)
        result = gram_schmidt_orthogonalize(v, basis)
        dot = (result @ basis).abs().item()
        assert dot < 1e-5, f"Expected orthogonal result, got dot={dot}"

    def test_removes_only_parallel_component(self):
        # v = basis + perp; orthogonalize should return perp
        basis = torch.tensor([1.0, 0.0, 0.0])
        perp = torch.tensor([0.0, 1.0, 0.0])
        v = basis + perp
        result = gram_schmidt_orthogonalize(v, basis)
        assert torch.allclose(result, perp, atol=1e-6)

    def test_already_orthogonal_unchanged(self):
        v = torch.tensor([0.0, 1.0, 0.0])
        basis = torch.tensor([1.0, 0.0, 0.0])
        result = gram_schmidt_orthogonalize(v, basis)
        assert torch.allclose(result, v, atol=1e-6)

    def test_parallel_vector_becomes_zero(self):
        basis = torch.tensor([1.0, 0.0, 0.0])
        v = 3.0 * basis
        result = gram_schmidt_orthogonalize(v, basis)
        assert result.norm().item() < 1e-5

    def test_output_shape_preserved(self):
        v = torch.randn(256)
        basis = torch.randn(256)
        result = gram_schmidt_orthogonalize(v, basis)
        assert result.shape == v.shape

    def test_zero_basis_does_not_crash(self):
        # Near-zero basis: denominator uses 1e-12 guard, result should equal v
        v = torch.tensor([1.0, 2.0, 3.0])
        basis = torch.zeros(3)
        result = gram_schmidt_orthogonalize(v, basis)
        assert torch.allclose(result, v, atol=1e-5)

    def test_high_dimensional_orthogonality(self):
        torch.manual_seed(42)
        v = torch.randn(2048)
        basis = torch.randn(2048)
        result = gram_schmidt_orthogonalize(v, basis)
        cos_sim = (result @ basis) / (result.norm() * basis.norm() + 1e-12)
        assert cos_sim.abs().item() < 1e-5


class TestRevisVectorMetadata:
    """Test compute_revis_vector using stub forward functions to avoid model loading."""

    def _make_stub_model_and_processor(self, hidden_dim: int, n_layers: int, seed: int = 0):
        """Return a minimal namespace that satisfies compute_revis_vector's interface
        by monkey-patching forward_with_hidden_states inside the module."""
        return hidden_dim, n_layers, seed

    def test_orthogonality_after_pipeline(self, monkeypatch):
        """compute_revis_vector should produce a vector orthogonal to v_lang."""
        hidden_dim = 64
        torch.manual_seed(0)

        # Patch the two inner functions so compute_revis_vector runs without a model
        import revis.vector_calculator as rvc

        def fake_visual_dir(model, processor, samples, layer_idx, device, token_position):
            return torch.randn(hidden_dim)

        def fake_lang_dir(model, processor, samples, layer_idx, device, token_position):
            return torch.randn(hidden_dim)

        monkeypatch.setattr(rvc, "compute_visual_direction", fake_visual_dir)
        monkeypatch.setattr(rvc, "compute_language_prior_direction", fake_lang_dir)

        v_pure, meta = rvc.compute_revis_vector(
            model=None, processor=None, samples=[{}] * 5,
            layer_idx=10, device="cpu", normalize=False,
        )

        # After Gram-Schmidt, v_pure must be orthogonal to v_lang
        # Re-derive v_lang to check (same seed, deterministic)
        torch.manual_seed(0)
        torch.randn(hidden_dim)   # consume visual direction call
        v_lang = torch.randn(hidden_dim)
        dot = (v_pure @ v_lang).abs().item()
        assert dot < 1e-4, f"v_pure not orthogonal to v_lang: dot={dot}"

    def test_normalized_vector_has_unit_norm(self, monkeypatch):
        hidden_dim = 128
        import revis.vector_calculator as rvc

        monkeypatch.setattr(rvc, "compute_visual_direction",
                            lambda *a, **kw: torch.randn(hidden_dim))
        monkeypatch.setattr(rvc, "compute_language_prior_direction",
                            lambda *a, **kw: torch.randn(hidden_dim))

        v_pure, _ = rvc.compute_revis_vector(
            model=None, processor=None, samples=[{}],
            layer_idx=5, device="cpu", normalize=True,
        )
        assert abs(v_pure.norm().item() - 1.0) < 1e-5

    def test_metadata_keys_present(self, monkeypatch):
        import revis.vector_calculator as rvc

        monkeypatch.setattr(rvc, "compute_visual_direction",
                            lambda *a, **kw: torch.randn(64))
        monkeypatch.setattr(rvc, "compute_language_prior_direction",
                            lambda *a, **kw: torch.randn(64))

        _, meta = rvc.compute_revis_vector(
            model=None, processor=None, samples=[{}],
            layer_idx=22, device="cpu",
        )
        for key in ("layer_idx", "cos_visual_lang_before", "cos_visual_lang_after",
                    "v_visual_norm", "v_lang_norm", "v_pure_norm", "n_samples"):
            assert key in meta, f"Missing metadata key: {key}"

    def test_cos_after_is_smaller_than_before(self, monkeypatch):
        """Orthogonalization should reduce (ideally zero) cosine similarity."""
        import revis.vector_calculator as rvc
        torch.manual_seed(7)

        # Make visual direction partially aligned with language direction
        v_lang_fixed = torch.randn(64)
        v_vis_fixed = v_lang_fixed * 0.5 + torch.randn(64) * 0.5

        monkeypatch.setattr(rvc, "compute_visual_direction",
                            lambda *a, **kw: v_vis_fixed.clone())
        monkeypatch.setattr(rvc, "compute_language_prior_direction",
                            lambda *a, **kw: v_lang_fixed.clone())

        _, meta = rvc.compute_revis_vector(
            model=None, processor=None, samples=[{}],
            layer_idx=0, device="cpu", normalize=False,
        )
        assert abs(meta["cos_visual_lang_after"]) < abs(meta["cos_visual_lang_before"]), (
            "Orthogonalization should reduce cosine similarity with language prior"
        )
