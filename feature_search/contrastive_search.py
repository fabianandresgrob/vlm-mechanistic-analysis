"""
Exp 3.1: Contrastive SAE Feature Search

For each SAE latent j, computes:
    f_vis_j   = fraction of samples where latent j activates under condition B (vis)
    f_blind_j = fraction of samples where latent j activates under condition A (blind)
    s_visual_j = f_vis_j - f_blind_j  (visual reliance latents, positive)
    s_prior_j  = f_blind_j - f_vis_j  (language prior latents, positive)

Noise filter: exclude latents with f_vis > 0.02 on a random non-visual text baseline
(following Ferrando et al. — these are high-frequency "noise" features).

Reference: Ferrando et al. (ICLR 2025) "Do I Know This Entity?"
"""
from __future__ import annotations
import logging
import numpy as np
import torch
from feature_search.sae_utils import load_sae

logger = logging.getLogger(__name__)


def compute_activation_frequencies(
    acts: np.ndarray,
    sae,
    batch_size: int = 64,
) -> np.ndarray:
    """Compute per-latent activation frequency over a set of hidden states.

    A latent is considered "active" if its post-TopK activation value > 0.

    Args:
        acts: Hidden states, shape (n_samples, d_model).
        sae: Loaded SAELens SAE object.
        batch_size: Process this many samples at once to fit in VRAM.

    Returns:
        freq: np.ndarray of shape (d_sae,) with values in [0, 1].
    """
    n = acts.shape[0]
    d_sae = sae.cfg.d_sae
    total_active = np.zeros(d_sae, dtype=np.float64)

    for start in range(0, n, batch_size):
        chunk = acts[start:start + batch_size]
        x = torch.tensor(chunk, dtype=torch.float32, device=sae.device)
        x = x.unsqueeze(1)   # (batch, 1, d_model)
        feat = sae.encode(x)  # (batch, 1, d_sae)
        active = (feat[:, 0, :] > 0).float().cpu().numpy()   # (batch, d_sae)
        total_active += active.sum(axis=0)

    return total_active / n


def separation_scores(
    acts_vis: np.ndarray,
    acts_blind: np.ndarray,
    sae,
    noise_baseline_acts: np.ndarray | None = None,
    noise_threshold: float = 0.02,
    batch_size: int = 64,
) -> dict:
    """Compute visual-reliance and language-prior separation scores for all latents.

    Args:
        acts_vis:   Hidden states under condition B (vis). Shape (n, d_model).
        acts_blind: Hidden states under condition A (blind). Shape (n, d_model).
        sae:        Loaded SAELens SAE.
        noise_baseline_acts: Optional text-only hidden states for noise filtering.
                             Shape (n_baseline, d_model). If None, no noise filter.
        noise_threshold: Latents with freq > this on noise_baseline are excluded.
        batch_size: Encoding batch size.

    Returns:
        dict with keys:
            'f_vis':       np.ndarray (d_sae,) — activation freq under vis
            'f_blind':     np.ndarray (d_sae,) — activation freq under blind
            's_visual':    np.ndarray (d_sae,) — f_vis - f_blind
            's_prior':     np.ndarray (d_sae,) — f_blind - f_vis
            'noise_mask':  np.ndarray (d_sae,) bool — True if latent passes noise filter
            'top_visual':  np.ndarray (k,) — indices of top visual reliance latents
            'top_prior':   np.ndarray (k,) — indices of top language prior latents
    """
    logger.info("Computing vis activation frequencies (%d samples)…", len(acts_vis))
    f_vis = compute_activation_frequencies(acts_vis, sae, batch_size)

    logger.info("Computing blind activation frequencies (%d samples)…", len(acts_blind))
    f_blind = compute_activation_frequencies(acts_blind, sae, batch_size)

    s_visual = f_vis - f_blind
    s_prior = f_blind - f_vis

    # Noise filter
    noise_mask = np.ones(sae.cfg.d_sae, dtype=bool)
    if noise_baseline_acts is not None:
        logger.info("Computing noise baseline frequencies (%d samples)…",
                    len(noise_baseline_acts))
        f_noise = compute_activation_frequencies(noise_baseline_acts, sae, batch_size)
        noise_mask = f_noise <= noise_threshold
        n_filtered = (~noise_mask).sum()
        logger.info("Noise filter removed %d / %d latents (freq > %.2f).",
                    n_filtered, sae.cfg.d_sae, noise_threshold)

    # Rank (only among non-noise latents)
    masked_s_visual = np.where(noise_mask, s_visual, -np.inf)
    masked_s_prior = np.where(noise_mask, s_prior, -np.inf)

    top_visual = np.argsort(masked_s_visual)[::-1][:100]
    top_prior = np.argsort(masked_s_prior)[::-1][:100]

    return {
        "f_vis": f_vis,
        "f_blind": f_blind,
        "s_visual": s_visual,
        "s_prior": s_prior,
        "noise_mask": noise_mask,
        "top_visual": top_visual,
        "top_prior": top_prior,
    }
