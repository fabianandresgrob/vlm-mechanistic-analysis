"""
Exp 2.4: Three-condition contrastive forward pass.

For each sample, runs up to three forward passes through the model:
  Condition A (blind):          No image (zeroed pixel_values). Pure language prior.
  Condition B (original):       Original image. Vision and prior may agree.
  Condition C (counterfactual): Counterfactual image. Vision contradicts prior.

At each LLM layer, extracts the hidden state of the last input token.
These per-layer hidden states are the input to VIP detection and TVI computation.

Reference: Long et al. (ICLR 2026) "Understanding Language Prior of LVLMs by
Contrasting Chain-of-Embedding" — adapted to Gemma 3 and extended to 3 conditions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from chain_of_embedding.models.gemma3 import (
    forward_with_hidden_states,
    num_llm_layers,
)

logger = logging.getLogger(__name__)


@dataclass
class ContrastiveSample:
    """Input for a three-condition forward pass.

    Attributes:
        messages:       Chat-format messages (with image placeholder in content).
                        The same messages are used for all three conditions.
        image:          PIL Image for condition B (original). Required.
        cf_image:       PIL Image for condition C (counterfactual). Optional —
                        if None, only conditions A and B are run.
        id:             Sample identifier for result tracking.
        answer:         Ground truth answer string (optional).
        is_correct_vis: Whether condition B yields the correct answer (set after
                        running greedy decode, used for D_VT/D_T split).
    """
    messages: list[dict]
    image: object                        # PIL Image
    cf_image: Optional[object] = None   # PIL Image or None
    id: object = None
    answer: str = ""
    is_correct_vis: Optional[bool] = None


@dataclass
class ContrastiveResult:
    """Per-layer hidden states from a three-condition forward pass.

    Hidden states are extracted at the LAST INPUT TOKEN position (the position
    that produces the next-token distribution), following Long et al.

    Attributes:
        sample_id:   Identifier from ContrastiveSample.
        hs_blind:    np.ndarray of shape [n_layers, hidden_dim]. Condition A.
        hs_vis:      np.ndarray of shape [n_layers, hidden_dim]. Condition B.
        hs_cf:       np.ndarray or None [n_layers, hidden_dim]. Condition C.
        n_layers:    Number of LLM decoder layers.
        has_cf:      Whether condition C was run.
        pred_blind:  Greedy-decoded answer token for condition A (optional).
        pred_vis:    Greedy-decoded answer token for condition B (optional).
        pred_cf:     Greedy-decoded answer token for condition C (optional).
    """
    sample_id: object
    hs_blind: "np.ndarray"
    hs_vis: "np.ndarray"
    hs_cf: Optional["np.ndarray"]
    n_layers: int
    has_cf: bool
    pred_blind: Optional[str] = None
    pred_vis: Optional[str] = None
    pred_cf: Optional[str] = None


def _build_inputs(sample: ContrastiveSample, processor, device: str) -> dict:
    """Build processor inputs from a ContrastiveSample (with image)."""
    text = processor.apply_chat_template(
        sample.messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = processor(
        text=text,
        images=[sample.image],
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in inputs.items()}


def _build_inputs_cf(sample: ContrastiveSample, processor, device: str) -> dict:
    """Build processor inputs using the counterfactual image."""
    text = processor.apply_chat_template(
        sample.messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = processor(
        text=text,
        images=[sample.cf_image],
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in inputs.items()}


@torch.no_grad()
def _greedy_next_token(model, inputs: dict, processor) -> str:
    """Run greedy decoding for a single next token. Returns the decoded string."""
    out = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
    )
    # out shape: (1, input_len + 1); take the last token
    new_token_id = out[0, -1].item()
    return processor.tokenizer.decode([new_token_id], skip_special_tokens=True).strip()


@torch.no_grad()
def run_contrastive_forward(
    model,
    processor,
    sample: ContrastiveSample,
    device: str = "cuda",
    decode_answers: bool = True,
) -> Optional[ContrastiveResult]:
    """Run up to three forward passes for a sample and collect per-layer hidden states.

    Args:
        model:          Loaded Gemma 3 model.
        processor:      Corresponding AutoProcessor.
        sample:         ContrastiveSample with image (and optionally cf_image).
        device:         Device string.
        decode_answers: If True, also run greedy decode for each condition to
                        get the predicted answer token (needed for D_VT/D_T split).

    Returns:
        ContrastiveResult, or None if the forward pass failed.
    """
    import numpy as np

    try:
        inputs_vis = _build_inputs(sample, processor, device)
    except Exception as e:
        logger.warning("Sample %s: failed to build inputs: %s", sample.id, e)
        return None

    # --- Condition B: original image ---
    try:
        _, hs_vis_all = forward_with_hidden_states(model, inputs_vis, include_image=True)
    except Exception as e:
        logger.warning("Sample %s: condition B forward failed: %s", sample.id, e)
        return None

    # --- Condition A: blind (zero out pixel_values) ---
    try:
        _, hs_blind_all = forward_with_hidden_states(model, inputs_vis, include_image=False)
    except Exception as e:
        logger.warning("Sample %s: condition A forward failed: %s", sample.id, e)
        return None

    n_layers = num_llm_layers(model)

    def _extract_last_token(hs_all: list) -> "np.ndarray":
        """Extract last-token hidden state from each layer. Shape: [n_layers, hidden_dim]."""
        layers = []
        for i in range(n_layers):
            h = hs_all[i + 1][0, -1, :]   # (hidden_dim,)
            layers.append(h.float().cpu().numpy())
        return np.stack(layers, axis=0)   # (n_layers, hidden_dim)

    hs_blind = _extract_last_token(hs_blind_all)
    hs_vis = _extract_last_token(hs_vis_all)

    # --- Condition C: counterfactual image (optional) ---
    hs_cf = None
    pred_cf = None
    has_cf = sample.cf_image is not None

    if has_cf:
        try:
            inputs_cf = _build_inputs_cf(sample, processor, device)
            _, hs_cf_all = forward_with_hidden_states(model, inputs_cf, include_image=True)
            hs_cf = _extract_last_token(hs_cf_all)
        except Exception as e:
            logger.warning("Sample %s: condition C forward failed: %s", sample.id, e)
            has_cf = False

    # --- Greedy decoding (for D_VT/D_T split) ---
    pred_blind = pred_vis = None
    if decode_answers:
        try:
            pred_vis = _greedy_next_token(model, inputs_vis, processor)
            blind_inputs = dict(inputs_vis)
            blind_inputs["pixel_values"] = torch.zeros_like(inputs_vis["pixel_values"])
            pred_blind = _greedy_next_token(model, blind_inputs, processor)
            if has_cf:
                pred_cf = _greedy_next_token(model, inputs_cf, processor)
        except Exception as e:
            logger.warning("Sample %s: greedy decode failed: %s", sample.id, e)

    return ContrastiveResult(
        sample_id=sample.id,
        hs_blind=hs_blind,
        hs_vis=hs_vis,
        hs_cf=hs_cf,
        n_layers=n_layers,
        has_cf=has_cf,
        pred_blind=pred_blind,
        pred_vis=pred_vis,
        pred_cf=pred_cf,
    )


def is_vision_dependent(result: ContrastiveResult, ground_truth: str = "") -> bool:
    """Classify a sample as vision-dependent (D_VT) or text-dependent (D_T).

    Following Long et al. Section 3.2: a sample is vision-dependent if the
    model's answer changes when the image is removed.

    If ground_truth is provided, also checks that the visual answer is correct
    (strict D_VT definition). If not provided, uses only the change criterion.
    """
    if result.pred_vis is None or result.pred_blind is None:
        return False
    answer_changed = result.pred_vis.lower() != result.pred_blind.lower()
    if not ground_truth:
        return answer_changed
    vis_correct = result.pred_vis.lower() == ground_truth.lower()
    return answer_changed and vis_correct
