"""
Total Visual Integration (TVI) computation.

TVI measures how much visual information is integrated into the model's
representations across the layers after the Visual Integration Point (VIP).

Definition (following Long et al., adapted):
    TVI = (1 / (n_layers - VIP)) * sum_{j=VIP}^{n_layers-1} cos_dist(Z_vis_j, Z_blind_j)

Normalised by hidden dimension for cross-scale comparison (Long et al. Figure 5):
    TVI_norm = TVI / sqrt(hidden_dim)

Key analyses:
    - D_VT/D_T split: compare TVI for vision-dependent vs text-dependent samples
    - Per-sample TVI correlates with correctness (Spearman ρ)
    - Average TVI vs model size is the core inverse scaling figure
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from chain_of_embedding.vip import cosine_distance


def compute_tvi(
    hs_blind: np.ndarray,
    hs_vis: np.ndarray,
    vip: int,
    normalize_by_dim: bool = True,
) -> float:
    """Compute Total Visual Integration for a single sample.

    Args:
        hs_blind:         Hidden states for condition A. Shape [n_layers, hidden_dim].
        hs_vis:           Hidden states for condition B. Shape [n_layers, hidden_dim].
        vip:              Visual Integration Point layer index (inclusive).
        normalize_by_dim: If True, divide by sqrt(hidden_dim) for cross-scale comparison.

    Returns:
        TVI scalar.
    """
    n_layers = hs_blind.shape[0]
    hidden_dim = hs_blind.shape[1]

    post_vip_layers = range(vip, n_layers)
    if not post_vip_layers:
        return 0.0

    tvi = np.mean([
        cosine_distance(hs_vis[j], hs_blind[j]) for j in post_vip_layers
    ])

    if normalize_by_dim:
        tvi = tvi / np.sqrt(hidden_dim)

    return float(tvi)


def compute_tvi_batch(
    hs_blind_batch: np.ndarray,
    hs_vis_batch: np.ndarray,
    vip: int,
    normalize_by_dim: bool = True,
) -> np.ndarray:
    """Compute TVI for a batch of samples.

    Args:
        hs_blind_batch: Shape [n_samples, n_layers, hidden_dim].
        hs_vis_batch:   Shape [n_samples, n_layers, hidden_dim].
        vip:            VIP layer index (same for all samples in batch).
        normalize_by_dim: Whether to normalise by sqrt(hidden_dim).

    Returns:
        TVI array of shape [n_samples].
    """
    return np.array([
        compute_tvi(hs_blind_batch[i], hs_vis_batch[i], vip, normalize_by_dim)
        for i in range(len(hs_blind_batch))
    ])


def compute_tvi_per_sample_vip(
    hs_blind_batch: np.ndarray,
    hs_vis_batch: np.ndarray,
    vips: list[int],
    normalize_by_dim: bool = True,
) -> np.ndarray:
    """Compute TVI using a per-sample VIP.

    Args:
        hs_blind_batch: Shape [n_samples, n_layers, hidden_dim].
        hs_vis_batch:   Shape [n_samples, n_layers, hidden_dim].
        vips:           Per-sample VIP layer indices. Length n_samples.
        normalize_by_dim: Whether to normalise by sqrt(hidden_dim).

    Returns:
        TVI array of shape [n_samples].
    """
    return np.array([
        compute_tvi(hs_blind_batch[i], hs_vis_batch[i], vips[i], normalize_by_dim)
        for i in range(len(hs_blind_batch))
    ])


def tvi_statistics(
    tvi_values: np.ndarray,
    is_correct: np.ndarray | None = None,
    is_vision_dependent: np.ndarray | None = None,
) -> dict:
    """Compute summary statistics and correlations for a set of TVI values.

    Args:
        tvi_values:          Shape [n_samples].
        is_correct:          Binary array [n_samples] or None.
        is_vision_dependent: Binary array [n_samples] (D_VT/D_T split) or None.

    Returns:
        dict with mean, std, median, and optional Spearman ρ / D_VT vs D_T stats.
    """
    stats: dict = {
        "mean": float(np.nanmean(tvi_values)),
        "std": float(np.nanstd(tvi_values)),
        "median": float(np.nanmedian(tvi_values)),
        "n": int(np.sum(~np.isnan(tvi_values))),
    }

    if is_correct is not None:
        mask = ~np.isnan(tvi_values) & ~np.isnan(is_correct)
        if mask.sum() >= 10:
            rho, pval = spearmanr(tvi_values[mask], is_correct[mask])
            stats["spearman_rho"] = float(rho)
            stats["spearman_pval"] = float(pval)

    if is_vision_dependent is not None:
        dvt_mask = is_vision_dependent.astype(bool) & ~np.isnan(tvi_values)
        dt_mask = ~is_vision_dependent.astype(bool) & ~np.isnan(tvi_values)
        stats["tvi_mean_dvt"] = float(np.nanmean(tvi_values[dvt_mask])) if dvt_mask.any() else float("nan")
        stats["tvi_mean_dt"] = float(np.nanmean(tvi_values[dt_mask])) if dt_mask.any() else float("nan")
        stats["n_dvt"] = int(dvt_mask.sum())
        stats["n_dt"] = int(dt_mask.sum())

    return stats
