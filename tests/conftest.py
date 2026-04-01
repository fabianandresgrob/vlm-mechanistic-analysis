"""
Shared fixtures and pytest configuration.

Unit tests run with no model — just numpy/torch tensors.
Integration tests (marked @pytest.mark.integration) require:
    pytest --model google/gemma-3-4b-it --device mps

Device selection priority: cuda > mps > cpu
"""
from __future__ import annotations

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# CLI options
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption("--model", default=None, help="HuggingFace model ID for integration tests")
    parser.addoption("--device", default=None, help="Force device (cuda/mps/cpu). Auto-detected if omitted.")
    parser.addoption("--download", action="store_true", default=False,
                     help="Run dataset download tests (requires internet; cached after first run)")


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: requires a real model (--model flag)")
    config.addinivalue_line("markers", "download: requires internet / HuggingFace dataset downloads (--download flag)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--model") is None:
        skip = pytest.mark.skip(reason="Pass --model <model_id> to run integration tests")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip)

    if not config.getoption("--download"):
        skip = pytest.mark.skip(reason="Pass --download to run dataset download tests")
        for item in items:
            if "download" in item.keywords:
                item.add_marker(skip)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def device(request) -> str:
    forced = request.config.getoption("--device")
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@pytest.fixture(scope="session")
def model_id(request) -> str | None:
    return request.config.getoption("--model")


@pytest.fixture(scope="session")
def loaded_model(model_id, device):
    """Load Gemma 3 once for the whole integration test session."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from chain_of_embedding.models.gemma3 import load_gemma3
    model, processor = load_gemma3(model_id, device=device)
    return model, processor


# ---------------------------------------------------------------------------
# Mock SAE — mimics SAELens SAE API without requiring HF download
# ---------------------------------------------------------------------------

class MockSAEConfig:
    def __init__(self, d_sae: int):
        self.d_sae = d_sae


class MockSAE:
    """Minimal mock matching the SAELens SAE interface used in contrastive_search and steering."""

    def __init__(self, d_model: int = 64, d_sae: int = 128, device: str = "cpu"):
        self.cfg = MockSAEConfig(d_sae)
        self.device = device
        # W_dec shape: (d_sae, d_model)
        self.W_dec = torch.randn(d_sae, d_model)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq, d_model) → (batch, seq, d_sae), TopK-sparse."""
        batch, seq, d_model = x.shape
        out = x @ self.W_dec.T   # (batch, seq, d_sae)
        # Simulate TopK sparsity: keep top-10, zero the rest
        k = min(10, self.cfg.d_sae)
        topk_vals, topk_idx = out.topk(k, dim=-1)
        sparse = torch.zeros_like(out)
        sparse.scatter_(-1, topk_idx, topk_vals.clamp(min=0))
        return sparse

    def decode(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (batch, seq, d_sae) → (batch, seq, d_model)"""
        return feat @ self.W_dec


@pytest.fixture
def mock_sae():
    return MockSAE(d_model=64, d_sae=128, device="cpu")


@pytest.fixture
def rng():
    return np.random.default_rng(42)
