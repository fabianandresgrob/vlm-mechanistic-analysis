"""
REVIS: Sparse Latent Steering via Gram-Schmidt Orthogonalization.

Adapted from Antgroup/REVIS (arXiv:2602.11824) for Gemma 3.

The core idea: the raw visual direction (hidden state difference between
image-present and image-absent runs) is contaminated by language prior
components. Gram-Schmidt orthogonalization removes that contamination,
leaving a purified visual steering vector.

Two-step algorithm:
  1. v_visual = mean(h_with_image - h_without_image) at last-token position
  2. v_lang   = mean(h_without_image) at last-token position (language prior)
  3. v_pure   = v_visual - proj(v_visual onto v_lang)   [Gram-Schmidt step]

Reference: arXiv:2602.11824, Section 3.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from chain_of_embedding.models.gemma3 import forward_with_hidden_states

logger = logging.getLogger(__name__)


@torch.no_grad()
def compute_visual_direction(
    model,
    processor,
    samples: list[dict],
    layer_idx: int,
    device: str = "cuda",
    token_position: str = "last",
) -> torch.Tensor:
    """Compute mean(h_image - h_noimage) at a given layer.

    This is the raw (unorthogonalized) visual direction — how much the image
    shifts the hidden state at the last token of the prompt, on average.

    Args:
        model: Gemma3ForConditionalGeneration.
        processor: AutoProcessor.
        samples: List of dicts with 'image' (PIL Image) and 'messages'.
        layer_idx: LLM decoder layer to extract from (0-based).
        device: Device string.
        token_position: 'last' (recommended — generation token) or 'mean'.

    Returns:
        Tensor of shape (hidden_dim,).
    """
    diffs = []
    n_skipped = 0

    for sample in samples:
        text = processor.apply_chat_template(
            sample["messages"], add_generation_prompt=True, tokenize=False
        )
        image = sample.get("image")

        # Build inputs with image
        try:
            raw_img = processor(text=text, images=[image], return_tensors="pt")
            inputs_img = {k: v.to(device) for k, v in raw_img.items()}
        except Exception as e:
            logger.debug("Skipping sample (image input failed): %s", e)
            n_skipped += 1
            continue

        # Build inputs without image
        raw_txt = processor(text=text, images=None, return_tensors="pt")
        inputs_txt = {k: v.to(device) for k, v in raw_txt.items()}

        try:
            _, hs_img = forward_with_hidden_states(model, inputs_img, include_image=True)
            _, hs_txt = forward_with_hidden_states(model, inputs_txt, include_image=False)
        except Exception as e:
            logger.debug("Skipping sample (forward failed): %s", e)
            n_skipped += 1
            continue

        # hidden_states index: layer 0 = embedding, layer k+1 = after decoder k
        h_img = hs_img[layer_idx + 1][0]  # (seq_len, hidden_dim)
        h_txt = hs_txt[layer_idx + 1][0]

        if token_position == "last":
            # Use the last non-padding token (generation position)
            if "attention_mask" in inputs_img:
                last_idx = inputs_img["attention_mask"][0].nonzero()[-1].item()
            else:
                last_idx = h_img.shape[0] - 1
            # Text-only sequence may be shorter; use its own last token
            if "attention_mask" in inputs_txt:
                last_idx_txt = inputs_txt["attention_mask"][0].nonzero()[-1].item()
            else:
                last_idx_txt = h_txt.shape[0] - 1
            diff = h_img[last_idx].float() - h_txt[last_idx_txt].float()
        else:
            # Mean over all non-padding active tokens
            diff = h_img.float().mean(0) - h_txt.float().mean(0)

        diffs.append(diff)

    if not diffs:
        raise RuntimeError(
            f"No valid samples for visual direction computation "
            f"(skipped {n_skipped}/{len(samples)})."
        )
    logger.info(
        "Visual direction: %d samples used (%d skipped).", len(diffs), n_skipped
    )
    return torch.stack(diffs).mean(0)  # (hidden_dim,)


@torch.no_grad()
def compute_language_prior_direction(
    model,
    processor,
    samples: list[dict],
    layer_idx: int,
    device: str = "cuda",
    token_position: str = "last",
) -> torch.Tensor:
    """Compute the language prior direction from text-only hidden states.

    The language prior direction is the mean last-token representation when
    running the model without any image. This captures what the model's
    residual stream 'defaults to' when relying purely on language statistics.

    Args:
        model: Gemma3ForConditionalGeneration.
        processor: AutoProcessor.
        samples: List of dicts with 'messages'.
        layer_idx: LLM decoder layer (0-based).
        device: Device string.
        token_position: 'last' or 'mean'.

    Returns:
        Tensor of shape (hidden_dim,).
    """
    vecs = []
    n_skipped = 0

    for sample in samples:
        text = processor.apply_chat_template(
            sample["messages"], add_generation_prompt=True, tokenize=False
        )
        raw_txt = processor(text=text, images=None, return_tensors="pt")
        inputs_txt = {k: v.to(device) for k, v in raw_txt.items()}

        try:
            _, hs_txt = forward_with_hidden_states(model, inputs_txt, include_image=False)
        except Exception as e:
            logger.debug("Skipping sample (forward failed): %s", e)
            n_skipped += 1
            continue

        h_txt = hs_txt[layer_idx + 1][0]  # (seq_len, hidden_dim)
        if token_position == "last":
            if "attention_mask" in inputs_txt:
                last_idx = inputs_txt["attention_mask"][0].nonzero()[-1].item()
            else:
                last_idx = h_txt.shape[0] - 1
            vecs.append(h_txt[last_idx].float())
        else:
            vecs.append(h_txt.float().mean(0))

    if not vecs:
        raise RuntimeError(
            f"No valid samples for language prior direction "
            f"(skipped {n_skipped}/{len(samples)})."
        )
    logger.info(
        "Language prior direction: %d samples used (%d skipped).", len(vecs), n_skipped
    )
    return torch.stack(vecs).mean(0)  # (hidden_dim,)


def gram_schmidt_orthogonalize(
    v: torch.Tensor, basis: torch.Tensor
) -> torch.Tensor:
    """Remove the component of v along basis (single Gram-Schmidt step).

    v_pure = v - (v·basis / ||basis||²) * basis

    Args:
        v: Vector to orthogonalize, shape (d,).
        basis: Direction to subtract out, shape (d,).

    Returns:
        Orthogonalized vector, shape (d,). Not normalized.
    """
    projection = (v @ basis) / (basis @ basis + 1e-12) * basis
    return v - projection


def compute_revis_vector(
    model,
    processor,
    samples: list[dict],
    layer_idx: int,
    device: str = "cuda",
    normalize: bool = True,
    token_position: str = "last",
) -> tuple[torch.Tensor, dict]:
    """Full REVIS pipeline: purified visual steering vector.

    Computes the raw visual direction (image - no-image mean difference),
    orthogonalizes it against the language prior direction to remove
    contamination, and optionally unit-normalizes.

    Args:
        model: Gemma3ForConditionalGeneration.
        processor: AutoProcessor.
        samples: Calibration samples — dicts with 'image' and 'messages'.
        layer_idx: Layer to compute vectors at.
        device: Device string.
        normalize: If True, unit-normalize the final vector.
        token_position: 'last' (recommended) or 'mean'.

    Returns:
        (v_pure, metadata) where metadata contains intermediate vectors and
        diagnostics (cosine similarity before/after orthogonalization, norms).
    """
    v_visual = compute_visual_direction(
        model, processor, samples, layer_idx, device, token_position
    )
    v_lang = compute_language_prior_direction(
        model, processor, samples, layer_idx, device, token_position
    )

    cos_before = (
        (v_visual @ v_lang) / (v_visual.norm() * v_lang.norm() + 1e-12)
    ).item()

    v_pure = gram_schmidt_orthogonalize(v_visual, v_lang)

    cos_after = (
        (v_pure @ v_lang) / (v_pure.norm() * v_lang.norm() + 1e-12)
    ).item()

    logger.info(
        "Layer %d | cos(v_visual, v_lang) before=%.4f  after=%.4f",
        layer_idx, cos_before, cos_after,
    )

    if normalize:
        v_pure = v_pure / (v_pure.norm() + 1e-12)

    metadata = {
        "layer_idx": layer_idx,
        "v_visual_norm": v_visual.norm().item(),
        "v_lang_norm": v_lang.norm().item(),
        "v_pure_norm": v_pure.norm().item(),
        "cos_visual_lang_before": cos_before,
        "cos_visual_lang_after": cos_after,
        "n_samples": len(samples),
        "token_position": token_position,
        "normalized": normalize,
    }

    return v_pure, metadata
