"""Tests for eva/eva_decoding.py — pure logic, no model required."""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch

from eva.eva_decoding import accuracy_summary, eva_decode_sample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VOCAB = 32
SEQ_LEN = 10
HIDDEN = 64
N_LAYERS = 4  # decoder layers; hidden_states has N_LAYERS+1 entries


def _make_inputs(seq_len: int = SEQ_LEN) -> dict:
    return {
        "input_ids": torch.zeros(1, seq_len, dtype=torch.long),
        "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
        "pixel_values": torch.zeros(1, 3, 224, 224),
    }


def _fake_hidden_states(seq_len: int) -> list[torch.Tensor]:
    """N_LAYERS+1 hidden state tensors of shape (1, seq_len, HIDDEN)."""
    return [torch.randn(1, seq_len, HIDDEN) for _ in range(N_LAYERS + 1)]


class TestEvaDecodeAttentionMaskGrowth:
    """Verify that attention_mask stays in sync with input_ids as tokens are generated.

    This is a regression test for the bug where the EVA autoregressive loop grew
    input_ids but left attention_mask at the original length, causing a shape
    mismatch on the second forward pass.
    """

    def test_attention_mask_matches_input_ids_every_step(self, monkeypatch):
        """Capture input shapes on every forward_with_hidden_states call and assert
        attention_mask.shape[1] == input_ids.shape[1] at every step."""
        import eva.eva_decoding as eva_mod

        observed_shapes = []  # list of (input_ids_len, attention_mask_len) per call

        def fake_forward(model, inputs, include_image=True):
            ids_len = inputs["input_ids"].shape[1]
            mask_len = inputs["attention_mask"].shape[1] if "attention_mask" in inputs else ids_len
            observed_shapes.append((ids_len, mask_len))
            seq_len = ids_len
            hs = _fake_hidden_states(seq_len)
            logits = torch.randn(1, seq_len, VOCAB)
            return logits, hs

        def fake_early_exit(model, hidden_state):
            return torch.randn(hidden_state.shape[0], VOCAB)

        # Fake processor
        class FakeTokenizer:
            eos_token_id = 99  # never emit EOS so we run all max_new_tokens steps

            def decode(self, ids, skip_special_tokens=True):
                return "fake"

        class FakeProcessor:
            tokenizer = FakeTokenizer()

            def apply_chat_template(self, messages, **kwargs):
                return "prompt"

            def __call__(self, text, images, return_tensors):
                return _make_inputs(SEQ_LEN)

        # Fake model.generate returns SEQ_LEN + 1 token sequence
        class FakeModel:
            def generate(self, **kwargs):
                input_ids = kwargs["input_ids"]
                extra = torch.zeros(1, 1, dtype=torch.long)
                return torch.cat([input_ids, extra], dim=1)

        monkeypatch.setattr(eva_mod, "forward_with_hidden_states", fake_forward)
        monkeypatch.setattr(eva_mod, "early_exit_logits", fake_early_exit)

        max_new_tokens = 4
        sample = {
            "messages": [{"role": "user", "content": "test"}],
            "image": None,
        }

        eva_decode_sample(
            FakeModel(), FakeProcessor(), sample,
            target_layer=1, alpha=1.0, device="cpu",
            max_new_tokens=max_new_tokens,
        )

        # Every forward call should have matching lengths
        assert len(observed_shapes) > 0
        for step, (ids_len, mask_len) in enumerate(observed_shapes):
            assert ids_len == mask_len, (
                f"Step {step}: input_ids length {ids_len} != "
                f"attention_mask length {mask_len} (regression: attention_mask not extended)"
            )

    def test_input_ids_grows_by_one_per_step(self, monkeypatch):
        """input_ids should grow by exactly 1 token per generated token."""
        import eva.eva_decoding as eva_mod

        ids_lengths = []

        def fake_forward(model, inputs, include_image=True):
            ids_lengths.append(inputs["input_ids"].shape[1])
            seq_len = inputs["input_ids"].shape[1]
            hs = _fake_hidden_states(seq_len)
            logits = torch.randn(1, seq_len, VOCAB)
            return logits, hs

        monkeypatch.setattr(eva_mod, "forward_with_hidden_states", fake_forward)
        monkeypatch.setattr(eva_mod, "early_exit_logits",
                            lambda model, h: torch.randn(h.shape[0], VOCAB))

        class FakeTokenizer:
            eos_token_id = 99

            def decode(self, ids, skip_special_tokens=True):
                return "fake"

        class FakeProcessor:
            tokenizer = FakeTokenizer()

            def apply_chat_template(self, messages, **kwargs):
                return "prompt"

            def __call__(self, text, images, return_tensors):
                return _make_inputs(SEQ_LEN)

        class FakeModel:
            def generate(self, **kwargs):
                return kwargs["input_ids"]

        monkeypatch.setattr(eva_mod, "forward_with_hidden_states", fake_forward)
        monkeypatch.setattr(eva_mod, "early_exit_logits",
                            lambda model, h: torch.randn(h.shape[0], VOCAB))

        max_new_tokens = 3
        eva_decode_sample(
            FakeModel(), FakeProcessor(),
            {"messages": [], "image": None},
            target_layer=1, alpha=1.0, device="cpu",
            max_new_tokens=max_new_tokens,
        )

        # ids_lengths has 2 entries per step (vis + blind passes)
        # Each pair of entries should have the same length, and length should
        # increase by 1 after each complete step.
        step_lengths = ids_lengths[::2]  # one per step (vis pass)
        for i in range(1, len(step_lengths)):
            assert step_lengths[i] == step_lengths[i - 1] + 1, (
                f"Step {i}: expected length {step_lengths[i-1]+1}, got {step_lengths[i]}"
            )


class TestAccuracySummary:
    def test_all_correct(self):
        results = [
            {"is_correct_vanilla": True, "is_correct_eva": True},
            {"is_correct_vanilla": True, "is_correct_eva": True},
        ]
        s = accuracy_summary(results)
        assert s["vanilla_accuracy"] == 1.0
        assert s["eva_accuracy"] == 1.0
        assert s["delta"] == 0.0

    def test_eva_improves(self):
        results = [
            {"is_correct_vanilla": False, "is_correct_eva": True},
            {"is_correct_vanilla": False, "is_correct_eva": False},
        ]
        s = accuracy_summary(results)
        assert s["vanilla_accuracy"] == 0.0
        assert s["eva_accuracy"] == 0.5
        assert s["delta"] == pytest.approx(0.5)

    def test_no_labels_returns_error(self):
        results = [{"vanilla_answer": "yes", "eva_answer": "no"}]
        s = accuracy_summary(results)
        assert "error" in s
