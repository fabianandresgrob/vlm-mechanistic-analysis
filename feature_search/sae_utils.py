"""
Shared SAE utilities for WS3 feature search and steering.

Provides:
- load_sae(): load a GemmaScope 2 IT SAE, with process-level caching
- extract_answer_token_acts(): extract hidden states at the last input token
  position for a list of samples, running two conditions (vis / blind)
"""
from __future__ import annotations
import logging
from typing import Optional
import torch
from sae_lens import SAE
from chain_of_embedding.models.gemma3 import forward_with_hidden_states

logger = logging.getLogger(__name__)

_sae_cache: dict = {}


def load_sae(
    layer_idx: int,
    model_size: str = "4b",
    width: str = "16k",
    l0_level: str = "medium",
    device: str = "cuda",
) -> Optional[SAE]:
    """Load a GemmaScope 2 IT residual-stream SAE for a given layer.

    Uses the resid_post_all release (all-layers variant).
    Returns None if the layer is unavailable.

    Args:
        layer_idx: LLM decoder layer (0-based).
        model_size: '270m', '1b', '4b', '12b', '27b'.
        width: '16k', '64k', '256k', '1m'.
        l0_level: 'small', 'medium', 'large'.
        device: torch device string.
    """
    key = (layer_idx, model_size, width, l0_level)
    if key in _sae_cache:
        return _sae_cache[key]

    release = f"gemma-scope-2-{model_size}-it-resid_post_all"
    sae_id = f"layer_{layer_idx}_width_{width}_l0_{l0_level}"
    try:
        sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
        sae.eval()
        _sae_cache[key] = sae
        logger.debug("Loaded SAE %s / %s", release, sae_id)
        return sae
    except Exception as e:
        logger.warning("SAE unavailable for layer %d: %s", layer_idx, e)
        _sae_cache[key] = None
        return None


def clear_sae_cache() -> None:
    """Free all cached SAEs."""
    _sae_cache.clear()


@torch.no_grad()
def extract_answer_token_acts(
    model,
    processor,
    samples: list[dict],
    layer_idx: int,
    device: str = "cuda",
) -> tuple["np.ndarray", "np.ndarray"]:
    """Extract hidden states at the last input token for two conditions.

    For each sample, runs two forward passes:
      - Condition B (vis): original image
      - Condition A (blind): zeroed pixel_values

    Extracts hidden state at hidden_states[layer_idx + 1][:, -1, :] (last token).

    Args:
        model: Loaded Gemma3ForConditionalGeneration.
        processor: Corresponding AutoProcessor.
        samples: List of dicts with 'image' (PIL) and 'messages'.
        layer_idx: Which LLM layer to extract from (0-based).
        device: Device string.

    Returns:
        (acts_vis, acts_blind): np.ndarray arrays of shape (n_valid, d_model).
        Samples that fail are silently skipped; caller should check n_valid.
    """
    import numpy as np
    from tqdm import tqdm

    vis_list, blind_list = [], []

    for sample in tqdm(samples, desc=f"Extracting layer {layer_idx} activations"):
        try:
            text = processor.apply_chat_template(
                sample["messages"], add_generation_prompt=True, tokenize=False
            )
            raw = processor(text=text, images=[sample["image"]], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in raw.items()}
        except Exception as e:
            logger.warning("Sample %s: input build failed: %s", sample.get("id"), e)
            continue

        try:
            _, hs_vis = forward_with_hidden_states(model, inputs, include_image=True)
            _, hs_blind = forward_with_hidden_states(model, inputs, include_image=False)
        except Exception as e:
            logger.warning("Sample %s: forward pass failed: %s", sample.get("id"), e)
            continue

        vis_list.append(hs_vis[layer_idx + 1][0, -1, :].float().cpu().numpy())
        blind_list.append(hs_blind[layer_idx + 1][0, -1, :].float().cpu().numpy())

    if not vis_list:
        raise RuntimeError("No samples successfully extracted.")

    return np.stack(vis_list), np.stack(blind_list)
