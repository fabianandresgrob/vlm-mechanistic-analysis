"""
Exp 2.1: EVA JS Divergence Analysis

Replicates Zhou et al. (EVA) methodology on Gemma 3.
Two-pass forward (with/without image), per-layer JS divergence via early exit.
Identifies the layer where visual information most influences token predictions.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from tqdm import tqdm

from chain_of_embedding.models.gemma3 import (
    early_exit_logits,
    forward_with_hidden_states,
    num_llm_layers,
)

logger = logging.getLogger(__name__)


def jensen_shannon_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Jensen-Shannon divergence (base-2 bits) between probability distributions.

    Args:
        p: Tensor of shape (..., vocab_size). Must be a valid probability distribution.
        q: Tensor of shape (..., vocab_size). Must be a valid probability distribution.
        eps: Small constant to avoid log(0).

    Returns:
        JSD in bits, shape (...,).
    """
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    m = 0.5 * (p + q)
    # KL(p || m) + KL(q || m) in nats, convert to bits by /ln(2)
    kl_pm = (p * (p.log() - m.log())).sum(dim=-1)
    kl_qm = (q * (q.log() - m.log())).sum(dim=-1)
    jsd = 0.5 * (kl_pm + kl_qm) / torch.log(torch.tensor(2.0, device=p.device))
    return jsd.clamp(min=0.0)  # numerical guard against tiny negatives


def _build_inputs(sample: dict, processor, device: str) -> dict:
    """Build processor inputs from a sample dict.

    Sample dict must have:
        - 'image': PIL Image or None
        - 'messages': list in Gemma 3 chat format (with image token in content)

    Returns processor output moved to `device`.
    """
    text = processor.apply_chat_template(
        sample["messages"],
        add_generation_prompt=True,
        tokenize=False,
    )
    images = [sample["image"]] if sample.get("image") is not None else None
    inputs = processor(text=text, images=images, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


@torch.no_grad()
def compute_layer_js_divergence(
    model,
    processor,
    samples: list[dict],
    target_token_position: str = "last",
    device: str = "cuda",
    batch_size: int = 1,
    output_dir: Optional[str] = None,
    resume: bool = False,
) -> dict:
    """Compute per-layer JS divergence for a list of samples.

    For each sample, performs two forward passes:
      1. Multimodal: with image (pixel_values as-is)
      2. Text-only:  image zeroed out (include_image=False)

    At each LLM layer, extracts hidden state at the target token position,
    applies early_exit_logits, softmax, then computes JS divergence.

    Args:
        model: Loaded Gemma3ForConditionalGeneration.
        processor: Corresponding AutoProcessor.
        samples: List of dicts with 'image' (PIL), 'messages', optionally 'is_correct'.
        target_token_position: 'last' (final generated token position, i.e. last input token).
        device: Device string.
        batch_size: Currently only 1 supported (variable sequence lengths).
        output_dir: If set, saves per-sample results there for resumability.
        resume: If True and output_dir set, skips already-processed samples.

    Returns:
        dict with keys:
            'js_per_layer':  np.ndarray of shape [num_samples, num_layers]
            'mean_js':       np.ndarray of shape [num_layers]
            'std_js':        np.ndarray of shape [num_layers]
            'is_correct':    np.ndarray of shape [num_samples] (NaN if not provided)
            'sample_ids':    list of length num_samples
    """
    n_layers = num_llm_layers(model)
    all_js: list[np.ndarray] = []
    all_correct: list[float] = []
    sample_ids: list = []

    # Determine already-processed sample IDs for resuming
    done_ids: set = set()
    if resume and output_dir:
        cache_path = os.path.join(output_dir, "js_per_layer.npz")
        if os.path.exists(cache_path):
            cached = np.load(cache_path, allow_pickle=True)
            done_ids = set(cached["sample_ids"].tolist())
            logger.info("Resuming: %d samples already cached.", len(done_ids))

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for idx, sample in enumerate(tqdm(samples, desc="EVA JS divergence")):
        sample_id = sample.get("id", idx)
        if sample_id in done_ids:
            continue

        try:
            inputs = _build_inputs(sample, processor, device)
        except Exception as e:
            logger.warning("Sample %s: failed to build inputs: %s", sample_id, e)
            continue

        try:
            _, hs_vis = forward_with_hidden_states(model, inputs, include_image=True)
            _, hs_blind = forward_with_hidden_states(model, inputs, include_image=False)
        except Exception as e:
            logger.warning("Sample %s: forward pass failed: %s", sample_id, e)
            continue

        js_layers = []
        for layer_i in range(n_layers):
            # hidden_states[0] = embedding; layer_i corresponds to hs[layer_i + 1]
            h_vis = hs_vis[layer_i + 1]    # (1, seq_len, hidden_dim)
            h_blind = hs_blind[layer_i + 1]

            if target_token_position == "last":
                h_vis = h_vis[:, -1, :]    # (1, hidden_dim)
                h_blind = h_blind[:, -1, :]
            else:
                raise ValueError(f"Unknown target_token_position: {target_token_position}")

            logits_vis = early_exit_logits(model, h_vis)     # (1, vocab_size)
            logits_blind = early_exit_logits(model, h_blind)

            p_vis = F.softmax(logits_vis, dim=-1)
            p_blind = F.softmax(logits_blind, dim=-1)

            jsd = jensen_shannon_divergence(p_vis, p_blind)  # (1,)
            js_layers.append(jsd.item())

        js_arr = np.array(js_layers, dtype=np.float32)  # (n_layers,)
        all_js.append(js_arr)
        all_correct.append(float(sample.get("is_correct", float("nan"))))
        sample_ids.append(sample_id)

    if not all_js:
        raise RuntimeError("No samples were successfully processed.")

    js_per_layer = np.stack(all_js, axis=0)          # (n_samples, n_layers)
    is_correct = np.array(all_correct, dtype=np.float32)
    mean_js = np.nanmean(js_per_layer, axis=0)
    std_js = np.nanstd(js_per_layer, axis=0)

    result = {
        "js_per_layer": js_per_layer,
        "mean_js": mean_js,
        "std_js": std_js,
        "is_correct": is_correct,
        "sample_ids": np.array(sample_ids, dtype=object),
    }

    if output_dir:
        np.savez(os.path.join(output_dir, "js_per_layer.npz"), **result)

    return result


def find_peak_layer(js_curve: np.ndarray) -> int:
    """Return the layer index with maximum mean JS divergence.

    Args:
        js_curve: 1-D array of JS divergence values per layer.

    Returns:
        Index of the peak layer.
    """
    return int(np.argmax(js_curve))


def correlate_with_correctness(
    js_per_layer: np.ndarray,
    is_correct: np.ndarray,
    layer_idx: int,
) -> tuple[float, float]:
    """Spearman correlation between JS divergence at a layer and sample correctness.

    Args:
        js_per_layer: Array of shape [n_samples, n_layers].
        is_correct: Binary array of shape [n_samples]. NaN values are dropped.
        layer_idx: Which layer's JS values to correlate.

    Returns:
        (rho, pvalue) Spearman correlation coefficient and p-value.
    """
    js_vals = js_per_layer[:, layer_idx]
    mask = ~np.isnan(is_correct) & ~np.isnan(js_vals)
    if mask.sum() < 10:
        logger.warning("Too few valid samples for correlation (%d).", mask.sum())
        return float("nan"), float("nan")
    rho, pvalue = spearmanr(js_vals[mask], is_correct[mask])
    return float(rho), float(pvalue)
