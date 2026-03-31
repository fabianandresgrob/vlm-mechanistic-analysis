"""
Exp 2.5: GemmaScope SAE Convergence Mapping

For each Gemma 3 LLM layer, loads the corresponding GemmaScope SAE and
measures how well it reconstructs visual vs text token representations.
The "convergence layer" is where visual tokens first become well-approximated
by the LLM's feature dictionary — this is the target layer range for WS3.

Expected finding: convergence around layer 18 (Venhoff et al. found ~18 in Gemma 2 2B).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from chain_of_embedding.models.gemma3 import (
    forward_with_hidden_states,
    get_visual_token_mask,
    num_llm_layers,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SAE loading
# ---------------------------------------------------------------------------

# Cache to avoid re-downloading SAEs within a run: {(layer, model_size, width): sae}
_sae_cache: dict = {}


def load_gemma_scope_sae(
    layer_idx: int,
    model_size: str = "4b",
    width: str = "16k",
    l0_level: str = "medium",
    hook_type: str = "resid_post",
    all_layers: bool = True,
    device: str = "cuda",
):
    """Load a GemmaScope 2 SAE for a specific layer via SAELens.

    GemmaScope 2 naming convention (google/gemma-scope-2-{size}-{pt|it}):
        release = "gemma-scope-2-{size}-pt-{hook_type}[_all]"
        sae_id  = "layer_{N}_width_{width}_l0_{l0_level}"

    Available widths: 16k, 64k, 256k, 1m  (64k/256k recommended)
    Available l0_level: "small" (10-20), "medium" (30-60), "large" (60-150)
    Available hook_type: "resid_post", "attn_out", "mlp_out", "transcoder"

    The `_all` suffix on the release name means SAEs for every layer are included.
    Without `_all`, only 4 depths are available (25%, 50%, 65%, 85% of model depth).
    For convergence profiling across all layers, use all_layers=True.

    Returns None if the requested layer has no SAE.

    Args:
        layer_idx: LLM decoder layer index (0-based).
        model_size: One of '270m', '1b', '4b', '12b', '27b'.
        width: Feature dictionary width. '16k' is smallest/fastest; '64k' recommended.
        l0_level: Sparsity level: 'small', 'medium', or 'large'.
        hook_type: SAE target: 'resid_post', 'attn_out', 'mlp_out', 'transcoder'.
        all_layers: If True, use the all-layers release variant.
        device: Device to load SAE onto.

    Returns:
        SAELens SAE object, or None if unavailable for this layer.
    """
    cache_key = (layer_idx, model_size, width, l0_level, hook_type, all_layers)
    if cache_key in _sae_cache:
        return _sae_cache[cache_key]

    try:
        from sae_lens import SAE
    except ImportError:
        raise ImportError(
            "sae-lens is required for Exp 2.5. Install with: uv pip install sae-lens"
        )

    suffix = f"{hook_type}_all" if all_layers else hook_type
    # Try IT first — we analyse IT model behaviour (lmms-eval was run on IT models)
    release_candidates = [
        f"gemma-scope-2-{model_size}-it-{suffix}",
        f"gemma-scope-2-{model_size}-pt-{suffix}",
    ]
    sae_id = f"layer_{layer_idx}_width_{width}_l0_{l0_level}"

    for release in release_candidates:
        try:
            sae = SAE.from_pretrained(
                release=release,
                sae_id=sae_id,
                device=device,
            )
            sae.eval()
            _sae_cache[cache_key] = sae
            logger.debug("Loaded SAE: %s / %s", release, sae_id)
            return sae
        except Exception:
            continue

    logger.warning(
        "No GemmaScope 2 SAE found for layer %d "
        "(model_size=%s, width=%s, l0=%s, hook=%s, all=%s). Skipping.",
        layer_idx, model_size, width, l0_level, hook_type, all_layers,
    )
    _sae_cache[cache_key] = None
    return None


def clear_sae_cache():
    """Free cached SAEs to reclaim GPU memory."""
    _sae_cache.clear()


# ---------------------------------------------------------------------------
# Per-layer reconstruction metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_sae_reconstruction_error(sae, hidden_states_layer: torch.Tensor) -> dict:
    """Compute SAE reconstruction quality metrics for a set of token representations.

    Args:
        sae: SAELens SAE object.
        hidden_states_layer: Float tensor of shape (n_tokens, hidden_dim).

    Returns:
        dict with keys: 'mse', 'normalized_mse', 'sparsity', 'mean_l0'.
    """
    x = hidden_states_layer.to(sae.device, dtype=sae.dtype if hasattr(sae, "dtype") else torch.float32)

    features = sae.encode(x)           # (n_tokens, n_features)
    x_hat = sae.decode(features)       # (n_tokens, hidden_dim)

    residual = x - x_hat
    mse = (residual ** 2).mean().item()
    mean_sq_norm = (x ** 2).mean().item()
    normalized_mse = mse / (mean_sq_norm + 1e-10)

    # L0: number of non-zero feature activations per token
    active = (features.abs() > 1e-6).float()
    mean_l0 = active.sum(dim=-1).mean().item()
    sparsity = mean_l0 / features.shape[-1]

    return {
        "mse": float(mse),
        "normalized_mse": float(normalized_mse),
        "sparsity": float(sparsity),
        "mean_l0": float(mean_l0),
    }


# ---------------------------------------------------------------------------
# Main convergence profiling
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_layer_convergence_profile(
    model,
    processor,
    samples: list[dict],
    model_size: str = "4b",
    layers: Optional[list[int]] = None,
    device: str = "cuda",
    batch_size: int = 1,
    width: str = "16k",
    l0_level: str = "medium",
    output_dir: Optional[str] = None,
    resume: bool = False,
) -> dict:
    """Profile GemmaScope 2 SAE reconstruction quality across layers.

    For each sample and each requested layer, extracts:
      - Visual token hidden states (positions of <image_soft_token> in the sequence)
      - Text token hidden states (all other non-padding positions)

    Then runs both through the layer's GemmaScope 2 SAE (resid_post) and records
    reconstruction error and sparsity metrics.

    GemmaScope 2 naming:
        release = "gemma-scope-2-{model_size}-pt-resid_post_all"
        sae_id  = "layer_{N}_width_{width}_l0_{l0_level}"

    Args:
        model: Loaded Gemma3ForConditionalGeneration.
        processor: Corresponding AutoProcessor.
        samples: List of dicts with 'image' (PIL Image) and 'messages'.
        model_size: One of '270m', '1b', '4b', '12b', '27b'.
        layers: Layer indices to probe. Defaults to every-other layer.
        device: Device string.
        batch_size: Only 1 supported currently.
        width: SAE width: '16k', '64k', '256k', or '1m'.
        l0_level: Sparsity level: 'small', 'medium', or 'large'.
        output_dir: If set, cache intermediate results here.
        resume: Skip computation if output already exists.

    Returns:
        dict with keys:
            'layers': list[int]
            'visual_mse', 'text_mse': np.ndarray shape [n_layers]
            'visual_normalized_mse', 'text_normalized_mse': np.ndarray
            'visual_sparsity', 'text_sparsity': np.ndarray
            'visual_mean_l0', 'text_mean_l0': np.ndarray
            'n_samples': int
    """
    from tqdm import tqdm

    n_total_layers = num_llm_layers(model)
    if layers is None:
        layers = list(range(0, n_total_layers, 2))

    # Check resume
    if resume and output_dir:
        cache_path = os.path.join(output_dir, "convergence_profile.npz")
        if os.path.exists(cache_path):
            logger.info("Loading cached convergence profile from %s", cache_path)
            data = np.load(cache_path, allow_pickle=True)
            return {k: data[k].tolist() if data[k].dtype == object else data[k] for k in data.files}

    # Accumulators: {layer_idx: {"visual": [...], "text": [...]}}
    layer_metrics: dict[int, dict[str, list]] = {
        li: {"visual": [], "text": []} for li in layers
    }

    # Pre-check which layers have SAEs available; avoid re-loading per sample
    available_layers = []
    for li in tqdm(layers, desc="Loading SAEs (pre-check)"):
        sae = load_gemma_scope_sae(
            li, model_size=model_size, width=width, l0_level=l0_level, device=device
        )
        if sae is not None:
            available_layers.append(li)
            # Free from cache; will re-load per-sample (too much VRAM if all cached)
            del sae
            _sae_cache.pop((li, model_size, width, l0_level, "resid_post", True), None)
            if device != "cpu":
                torch.cuda.empty_cache()

    if not available_layers:
        raise RuntimeError(
            "No GemmaScope SAEs could be loaded. Check model_size, width, and network."
        )
    logger.info(
        "SAEs available for %d / %d requested layers: %s",
        len(available_layers),
        len(layers),
        available_layers,
    )

    n_skipped = 0
    for sample in tqdm(samples, desc="Computing convergence profile"):
        # Build inputs
        try:
            from chain_of_embedding.models.gemma3 import _build_inputs_from_sample
            inputs = _build_inputs_from_sample(sample, processor, device)
        except ImportError:
            # Inline fallback
            text = processor.apply_chat_template(
                sample["messages"],
                add_generation_prompt=True,
                tokenize=False,
            )
            image = sample.get("image")
            images = [image] if image is not None else None
            raw = processor(text=text, images=images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in raw.items()}
        except Exception as e:
            logger.warning("Sample skipped (input build failed): %s", e)
            n_skipped += 1
            continue

        # Get visual token mask
        vis_mask = get_visual_token_mask(inputs, processor)  # (1, seq_len) or None
        if vis_mask is None or not vis_mask.any():
            logger.debug("No visual tokens found in sample — skipping.")
            n_skipped += 1
            continue

        # Full forward pass — extract all hidden states at once
        try:
            _, hidden_states = forward_with_hidden_states(model, inputs, include_image=True)
        except Exception as e:
            logger.warning("Forward pass failed: %s", e)
            n_skipped += 1
            continue

        # For each available layer, load SAE, compute metrics, free SAE
        for li in available_layers:
            hs_layer = hidden_states[li + 1][0]  # (seq_len, hidden_dim) — drop batch dim

            vis_idx = vis_mask[0].nonzero(as_tuple=True)[0]
            all_idx = torch.arange(hs_layer.shape[0], device=device)
            # Text tokens = non-visual, non-padding
            # Use attention_mask to exclude padding
            attn = inputs.get("attention_mask")
            if attn is not None:
                active_idx = attn[0].nonzero(as_tuple=True)[0]
            else:
                active_idx = all_idx
            text_idx = active_idx[~torch.isin(active_idx, vis_idx)]

            vis_hs = hs_layer[vis_idx]           # (n_vis, hidden_dim)
            text_hs = hs_layer[text_idx[:64]]    # cap at 64 text tokens per sample

            sae = load_gemma_scope_sae(
                li, model_size=model_size, width=width, l0_level=l0_level, device=device
            )
            if sae is None:
                continue

            vis_metrics = compute_sae_reconstruction_error(sae, vis_hs)
            text_metrics = compute_sae_reconstruction_error(sae, text_hs)

            layer_metrics[li]["visual"].append(vis_metrics)
            layer_metrics[li]["text"].append(text_metrics)

            # Free SAE immediately to save VRAM
            del sae
            _sae_cache.pop((li, model_size, width, l0_level, "resid_post", True), None)

        if device != "cpu":
            torch.cuda.empty_cache()

    logger.info("Processed %d samples (%d skipped).", len(samples) - n_skipped, n_skipped)

    # Aggregate
    def _agg(metrics_list: list[dict], key: str) -> float:
        vals = [m[key] for m in metrics_list if key in m]
        return float(np.mean(vals)) if vals else float("nan")

    result_layers = []
    vis_mse, txt_mse = [], []
    vis_nmse, txt_nmse = [], []
    vis_sp, txt_sp = [], []
    vis_l0, txt_l0 = [], []

    for li in layers:
        if li not in available_layers or not layer_metrics[li]["visual"]:
            continue
        result_layers.append(li)
        vis_m = layer_metrics[li]["visual"]
        txt_m = layer_metrics[li]["text"]
        vis_mse.append(_agg(vis_m, "mse"))
        txt_mse.append(_agg(txt_m, "mse"))
        vis_nmse.append(_agg(vis_m, "normalized_mse"))
        txt_nmse.append(_agg(txt_m, "normalized_mse"))
        vis_sp.append(_agg(vis_m, "sparsity"))
        txt_sp.append(_agg(txt_m, "sparsity"))
        vis_l0.append(_agg(vis_m, "mean_l0"))
        txt_l0.append(_agg(txt_m, "mean_l0"))

    profile = {
        "layers": np.array(result_layers, dtype=np.int32),
        "visual_mse": np.array(vis_mse),
        "text_mse": np.array(txt_mse),
        "visual_normalized_mse": np.array(vis_nmse),
        "text_normalized_mse": np.array(txt_nmse),
        "visual_sparsity": np.array(vis_sp),
        "text_sparsity": np.array(txt_sp),
        "visual_mean_l0": np.array(vis_l0),
        "text_mean_l0": np.array(txt_l0),
        "n_samples": np.array(len(samples) - n_skipped),
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        np.savez(os.path.join(output_dir, "convergence_profile.npz"), **profile)

    return profile


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def find_convergence_layer(profile: dict) -> int:
    """Find the layer where visual token reconstruction error first converges.

    Convergence criterion: normalized MSE < 0.1 for 3+ consecutive layers.
    Falls back to the layer with minimum normalized MSE.

    Args:
        profile: Output of compute_layer_convergence_profile().

    Returns:
        Layer index of the convergence point.
    """
    layers = np.array(profile["layers"])
    nmse = np.array(profile["visual_normalized_mse"])

    below = nmse < 0.1
    # Find first run of 3+ consecutive (in index space) layers below threshold
    for i in range(len(below) - 2):
        if below[i] and below[i + 1] and below[i + 2]:
            return int(layers[i])

    # Fallback: minimum
    return int(layers[np.nanargmin(nmse)])


def plot_convergence_profile(profile: dict, output_path: str) -> None:
    """Create a 2-panel figure showing SAE convergence across layers.

    Panel 1: Normalized MSE (visual vs text) with convergence layer marked.
    Panel 2: Sparsity (visual vs text).

    Args:
        profile: Output of compute_layer_convergence_profile().
        output_path: Path to save the PNG figure.
    """
    sns.set_theme(style="whitegrid")
    layers = np.array(profile["layers"])
    conv_layer = find_convergence_layer(profile)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Panel 1: Normalized MSE
    ax = axes[0]
    ax.plot(layers, profile["visual_normalized_mse"], "o-", label="Visual tokens", color="steelblue")
    ax.plot(layers, profile["text_normalized_mse"], "s--", label="Text tokens", color="coral")
    ax.axvline(conv_layer, color="green", linestyle=":", linewidth=1.5, label=f"Convergence (L{conv_layer})")
    ax.axhline(0.1, color="gray", linestyle="--", linewidth=1.0, alpha=0.6, label="Threshold 0.1")
    ax.set_xlabel("LLM Layer")
    ax.set_ylabel("Normalized MSE")
    ax.set_title("GemmaScope SAE Reconstruction Error")
    ax.legend(fontsize=9)

    # Panel 2: Sparsity
    ax = axes[1]
    ax.plot(layers, profile["visual_sparsity"], "o-", label="Visual tokens", color="steelblue")
    ax.plot(layers, profile["text_sparsity"], "s--", label="Text tokens", color="coral")
    ax.axvline(conv_layer, color="green", linestyle=":", linewidth=1.5, label=f"Convergence (L{conv_layer})")
    ax.set_xlabel("LLM Layer")
    ax.set_ylabel("Sparsity (fraction active features)")
    ax.set_title("SAE Feature Sparsity")
    ax.legend(fontsize=9)

    plt.suptitle(f"GemmaScope Convergence Profile  |  n={profile.get('n_samples', '?')} samples", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved convergence plot to %s", output_path)
