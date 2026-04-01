"""Tests for chain_of_embedding/models/gemma3.py helpers.

Focuses on get_visual_token_mask and the visual/text token separation logic
used in sae_convergence/convergence.py.

Unit tests (no model required) use a mocked processor and synthetic input_ids.
Integration tests (--model flag) use the real Gemma 3 processor + a small
synthetic image to verify the mask on actual tokenized output.
"""
from __future__ import annotations

import sys
import os

import pytest
import torch
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from chain_of_embedding.models.gemma3 import get_visual_token_mask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IMAGE_TOKEN_ID = 262144  # Gemma 3 <image_soft_token> token ID


def _mock_processor(image_token_id: int = IMAGE_TOKEN_ID):
    """Mock processor that returns image_token_id for the image token."""
    proc = MagicMock()
    proc.image_token = "<image_soft_token>"
    proc.tokenizer.convert_tokens_to_ids.return_value = image_token_id
    return proc


def _inputs(input_ids: list[int]) -> dict:
    return {"input_ids": torch.tensor([input_ids], dtype=torch.long)}


# ---------------------------------------------------------------------------
# get_visual_token_mask
# ---------------------------------------------------------------------------

class TestGetVisualTokenMask:
    def test_returns_true_at_image_positions(self):
        # Sequence: [text, image, image, text, text]
        ids = [1, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 2, 3]
        mask = get_visual_token_mask(_inputs(ids), _mock_processor())
        assert mask is not None
        expected = torch.tensor([[False, True, True, False, False]])
        assert mask.equal(expected)

    def test_returns_false_everywhere_for_text_only(self):
        ids = [1, 2, 3, 4]
        mask = get_visual_token_mask(_inputs(ids), _mock_processor())
        # All False — no image tokens
        assert mask is not None
        assert not mask.any()

    def test_returns_none_when_token_id_not_found(self):
        proc = _mock_processor()
        proc.tokenizer.convert_tokens_to_ids.return_value = None
        ids = [1, IMAGE_TOKEN_ID, 2]
        result = get_visual_token_mask(_inputs(ids), proc)
        assert result is None

    def test_returns_none_when_no_input_ids(self):
        result = get_visual_token_mask({}, _mock_processor())
        assert result is None

    def test_single_image_token(self):
        ids = [1, 2, IMAGE_TOKEN_ID, 3]
        mask = get_visual_token_mask(_inputs(ids), _mock_processor())
        assert mask is not None
        assert mask.sum().item() == 1
        assert mask[0, 2].item() is True

    def test_all_image_tokens(self):
        ids = [IMAGE_TOKEN_ID] * 5
        mask = get_visual_token_mask(_inputs(ids), _mock_processor())
        assert mask is not None
        assert mask.all()

    def test_shape_matches_input(self):
        ids = [1, IMAGE_TOKEN_ID, 2, IMAGE_TOKEN_ID, 3, 4]
        mask = get_visual_token_mask(_inputs(ids), _mock_processor())
        assert mask is not None
        assert mask.shape == (1, len(ids))


# ---------------------------------------------------------------------------
# Visual / text token separation (mirrors convergence.py lines 343-355)
# ---------------------------------------------------------------------------

class TestVisualTextSeparation:
    """Test that the index-splitting logic in convergence.py correctly
    partitions positions into visual vs text tokens."""

    def _split(self, input_ids: list[int], attention_mask: list[int] | None = None):
        """Reproduce the splitting logic from compute_layer_convergence_profile."""
        proc = _mock_processor()
        inputs = _inputs(input_ids)
        if attention_mask is not None:
            inputs["attention_mask"] = torch.tensor([attention_mask], dtype=torch.long)

        vis_mask = get_visual_token_mask(inputs, proc)
        assert vis_mask is not None

        seq_len = len(input_ids)
        device = "cpu"
        vis_idx = vis_mask[0].nonzero(as_tuple=True)[0]
        all_idx = torch.arange(seq_len, device=device)

        attn = inputs.get("attention_mask")
        if attn is not None:
            active_idx = attn[0].nonzero(as_tuple=True)[0]
        else:
            active_idx = all_idx

        text_idx = active_idx[~torch.isin(active_idx, vis_idx)]
        return vis_idx.tolist(), text_idx.tolist()

    def test_visual_and_text_are_disjoint(self):
        ids = [1, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 2, 3]
        vis, text = self._split(ids)
        assert set(vis) & set(text) == set()

    def test_visual_and_text_cover_all_active_positions(self):
        ids = [1, IMAGE_TOKEN_ID, 2, 3]
        vis, text = self._split(ids)
        assert sorted(vis + text) == list(range(len(ids)))

    def test_padding_excluded_from_text(self):
        # Position 4 is padding (attention_mask=0)
        ids = [1, IMAGE_TOKEN_ID, 2, 3, 0]
        attn = [1, 1, 1, 1, 0]
        vis, text = self._split(ids, attention_mask=attn)
        assert 4 not in text
        assert 4 not in vis

    def test_image_positions_excluded_from_text(self):
        ids = [10, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 20, 30]
        vis, text = self._split(ids)
        for pos in vis:
            assert ids[pos] == IMAGE_TOKEN_ID
        for pos in text:
            assert ids[pos] != IMAGE_TOKEN_ID

    def test_no_image_tokens_means_empty_visual(self):
        ids = [1, 2, 3, 4]
        proc = _mock_processor()
        inputs = _inputs(ids)
        vis_mask = get_visual_token_mask(inputs, proc)
        assert vis_mask is not None
        vis_idx = vis_mask[0].nonzero(as_tuple=True)[0]
        assert vis_idx.numel() == 0

    def test_count_with_mixed_sequence(self):
        # 3 image tokens, 4 text tokens
        ids = [1, IMAGE_TOKEN_ID, 2, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 3, 4]
        vis, text = self._split(ids)
        assert len(vis) == 3
        assert len(text) == 4


# ---------------------------------------------------------------------------
# Integration test: real processor + synthetic image
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestVisualTokenMaskRealProcessor:
    """Verify get_visual_token_mask on actual Gemma 3 processor output.

    Uses a small synthetic PIL image so no external data download is needed.
    Run with: pytest tests/test_gemma3_helpers.py --model google/gemma-3-4b-it
    """

    def _make_sample_image(self):
        """64x64 RGB image: red top half, blue bottom half."""
        from PIL import Image
        import numpy as np
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        arr[:32, :] = [255, 0, 0]   # red
        arr[32:, :] = [0, 0, 255]   # blue
        return Image.fromarray(arr)

    def test_mask_identifies_image_tokens(self, loaded_model):
        _, processor = loaded_model
        image = self._make_sample_image()
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "What colors do you see?"},
        ]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        raw = processor(text=text, images=[image], return_tensors="pt")

        mask = get_visual_token_mask(raw, processor)

        assert mask is not None, "Expected a mask but got None — image token ID not found"
        n_visual = mask.sum().item()
        assert n_visual > 0, "No visual tokens found — check processor.image_token"
        print(f"\n  Sequence length : {raw['input_ids'].shape[1]}")
        print(f"  Visual tokens   : {n_visual}")
        print(f"  Text tokens     : {raw['input_ids'].shape[1] - n_visual}")

    def test_visual_and_text_tokens_are_disjoint(self, loaded_model):
        _, processor = loaded_model
        image = self._make_sample_image()
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe the image."},
        ]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        raw = processor(text=text, images=[image], return_tensors="pt")

        mask = get_visual_token_mask(raw, processor)
        assert mask is not None

        vis_idx = mask[0].nonzero(as_tuple=True)[0]
        attn = raw.get("attention_mask")
        active_idx = attn[0].nonzero(as_tuple=True)[0] if attn is not None else torch.arange(raw["input_ids"].shape[1])
        text_idx = active_idx[~torch.isin(active_idx, vis_idx)]

        # Disjoint
        assert len(set(vis_idx.tolist()) & set(text_idx.tolist())) == 0

        # Visual positions hold the image token ID
        image_token_id = processor.tokenizer.convert_tokens_to_ids(
            processor.image_token if hasattr(processor, "image_token") else "<image_soft_token>"
        )
        for pos in vis_idx.tolist():
            assert raw["input_ids"][0, pos].item() == image_token_id, (
                f"Position {pos} flagged as visual but has token id "
                f"{raw['input_ids'][0, pos].item()}, expected {image_token_id}"
            )

        # Text positions do NOT hold the image token ID
        for pos in text_idx.tolist():
            assert raw["input_ids"][0, pos].item() != image_token_id, (
                f"Position {pos} flagged as text but has image token id"
            )

    def test_text_only_input_has_no_visual_tokens(self, loaded_model):
        _, processor = loaded_model
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "What is the capital of France?"},
        ]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        raw = processor(text=text, images=None, return_tensors="pt")

        mask = get_visual_token_mask(raw, processor)
        # Either None or all-False — no image tokens in a text-only input
        if mask is not None:
            assert not mask.any(), "Text-only input should have no visual tokens"
