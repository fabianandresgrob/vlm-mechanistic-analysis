"""
Integration tests — require a real model.

Run with:
    pytest tests/test_integration.py --model google/gemma-3-4b-it --device mps -v

Tests are skipped automatically when --model is not provided.

Design: tests use text-only inputs (no image) to avoid running the SigLIP vision
encoder, which is slow on MPS due to Metal shader compilation. The goal is to
validate the LLM layers and helper functions work correctly, not to benchmark
vision encoding speed.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def _text_only_inputs(processor, text: str, device: str) -> dict:
    """Build minimal text-only inputs (no image, no vision encoder)."""
    messages = [{"role": "user", "content": text}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=prompt, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


@pytest.mark.integration
class TestGemma3ModelLoader:

    def test_load_model(self, loaded_model):
        model, processor = loaded_model
        assert model is not None
        assert processor is not None

    def test_model_config(self, loaded_model):
        """Verify layer count and module accessors — no inference needed."""
        from chain_of_embedding.models.gemma3 import (
            get_final_norm,
            get_lm_head,
            num_llm_layers,
        )
        model, _ = loaded_model
        n = num_llm_layers(model)
        assert n > 0, "Expected at least one decoder layer"
        # Gemma 3 4b has 26 layers; any reasonable count is fine
        assert 10 <= n <= 50

        lm_head = get_lm_head(model)
        assert lm_head is not None

        norm = get_final_norm(model)
        assert norm is not None

    def test_forward_returns_correct_hidden_state_count(self, loaded_model, device):
        """Single text-only forward pass — checks hidden_states tuple length and shape."""
        from chain_of_embedding.models.gemma3 import (
            forward_with_hidden_states,
            num_llm_layers,
        )
        model, processor = loaded_model
        inputs = _text_only_inputs(processor, "What is 2+2?", device)

        n_layers = num_llm_layers(model)
        _, hs = forward_with_hidden_states(model, inputs, include_image=False)

        # embedding + n_layers decoder outputs
        assert len(hs) == n_layers + 1
        for h in hs:
            assert h.ndim == 3           # (batch, seq_len, hidden_dim)
            assert h.shape[0] == 1       # batch size 1
            assert h.shape[2] > 0        # hidden_dim

    def test_early_exit_logits_shape(self, loaded_model, device):
        """early_exit_logits produces (1, vocab_size) from a hidden state."""
        from chain_of_embedding.models.gemma3 import (
            early_exit_logits,
            forward_with_hidden_states,
        )
        model, processor = loaded_model
        inputs = _text_only_inputs(processor, "Yes or no?", device)

        _, hs = forward_with_hidden_states(model, inputs, include_image=False)
        h_last = hs[-1][:, -1, :]   # (1, hidden_dim)

        logits = early_exit_logits(model, h_last)
        # Use lm_head output size — may be padded beyond tokenizer.vocab_size
        lm_head_vocab_size = model.lm_head.weight.shape[0]
        assert logits.shape == (1, lm_head_vocab_size)

    def test_jsd_identical_passes_is_zero(self, loaded_model, device):
        """Two identical forward passes → JSD ≈ 0."""
        import torch.nn.functional as F
        from eva.js_divergence import jensen_shannon_divergence
        from chain_of_embedding.models.gemma3 import (
            early_exit_logits,
            forward_with_hidden_states,
        )
        model, processor = loaded_model
        inputs = _text_only_inputs(processor, "Answer: yes or no?", device)

        _, hs1 = forward_with_hidden_states(model, inputs, include_image=False)
        _, hs2 = forward_with_hidden_states(model, inputs, include_image=False)

        h1 = hs1[-1][:, -1, :]
        h2 = hs2[-1][:, -1, :]
        p1 = F.softmax(early_exit_logits(model, h1), dim=-1)
        p2 = F.softmax(early_exit_logits(model, h2), dim=-1)

        jsd = jensen_shannon_divergence(p1, p2).item()
        assert jsd == pytest.approx(0.0, abs=1e-4)

    def test_include_image_false_zeros_pixel_values(self, loaded_model, device):
        """include_image=False produces different hidden states than include_image=True
        when pixel_values are present (verifies the zeroing logic)."""
        from chain_of_embedding.models.gemma3 import forward_with_hidden_states
        from PIL import Image
        model, processor = loaded_model

        image = Image.new("RGB", (224, 224), color=(200, 100, 50))
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this."},
        ]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=prompt, images=[image], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        _, hs_vis = forward_with_hidden_states(model, inputs, include_image=True)
        _, hs_blind = forward_with_hidden_states(model, inputs, include_image=False)

        # At the last layer, visual and blind hidden states should differ
        h_vis = hs_vis[-1][:, -1, :].float().cpu()
        h_blind = hs_blind[-1][:, -1, :].float().cpu()
        assert not torch.allclose(h_vis, h_blind, atol=1e-3), \
            "Visual and blind hidden states should differ when image is present"
