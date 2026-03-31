"""
Visual Integration Point (VIP) detection.

The VIP is the layer at which visual information first begins to meaningfully
influence the model's internal representations — the point where the
chain-of-embedding "turns visual".

Definition (adapted from Long et al.):
    At each layer j, compute the cosine distance between the visual-condition
    hidden state and the blind-condition hidden state:
        d_vis_j  = cos_dist(Z_vis_j,  Z_blind_j)   [original image effect]
        d_cf_j   = cos_dist(Z_cf_j,   Z_blind_j)   [counterfactual image effect]
        d_disc_j = cos_dist(Z_vis_j,  Z_cf_j)       [discrimination between conditions]

    VIP = the first layer j where d_vis_j (or d_cf_j when available) first
    exceeds a threshold, computed as mean + k*std of the early-layer baseline.

    When counterfactual conditions are available, VIP is estimated from d_disc
    (model starts distinguishing original from counterfactual) as this is a
    cleaner signal than raw distance from blind.
"""

from __future__ import annotations

import numpy as np


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance (1 - cosine_similarity) between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(1.0 - np.dot(a, b) / (norm_a * norm_b))


def compute_layer_distances(
    hs_blind: np.ndarray,
    hs_vis: np.ndarray,
    hs_cf: np.ndarray | None = None,
) -> dict:
    """Compute per-layer cosine distances between conditions.

    Args:
        hs_blind: Array of shape [n_layers, hidden_dim]. Condition A.
        hs_vis:   Array of shape [n_layers, hidden_dim]. Condition B.
        hs_cf:    Array of shape [n_layers, hidden_dim] or None. Condition C.

    Returns:
        dict with keys:
            'd_vis':   np.ndarray [n_layers] — cos_dist(vis, blind)
            'd_cf':    np.ndarray [n_layers] or None — cos_dist(cf, blind)
            'd_disc':  np.ndarray [n_layers] or None — cos_dist(vis, cf)
    """
    n_layers = hs_blind.shape[0]

    d_vis = np.array([
        cosine_distance(hs_vis[i], hs_blind[i]) for i in range(n_layers)
    ])

    d_cf = None
    d_disc = None
    if hs_cf is not None:
        d_cf = np.array([
            cosine_distance(hs_cf[i], hs_blind[i]) for i in range(n_layers)
        ])
        d_disc = np.array([
            cosine_distance(hs_vis[i], hs_cf[i]) for i in range(n_layers)
        ])

    return {"d_vis": d_vis, "d_cf": d_cf, "d_disc": d_disc}


def detect_vip(
    d_vis: np.ndarray,
    d_cf: np.ndarray | None = None,
    d_disc: np.ndarray | None = None,
    baseline_layers: int = 4,
    threshold_k: float = 2.0,
    min_layer: int = 1,
) -> int:
    """Detect the Visual Integration Point (VIP) layer.

    Uses a threshold-based detection: the VIP is the first layer where the
    distance signal exceeds mean + k*std computed over the first `baseline_layers`.

    When counterfactual data is available, uses d_disc (original vs counterfactual
    discrimination) as the primary signal — it's more sensitive because it directly
    measures whether the model distinguishes visually different inputs.

    Falls back to d_vis (original vs blind) when d_disc is unavailable.

    Args:
        d_vis:           Per-layer cos_dist(vis, blind). Shape [n_layers].
        d_cf:            Per-layer cos_dist(cf, blind) or None.
        d_disc:          Per-layer cos_dist(vis, cf) or None.
        baseline_layers: Number of early layers used to estimate baseline stats.
        threshold_k:     Threshold = mean + k * std of baseline.
        min_layer:       Minimum layer index to consider as VIP (skip embedding).

    Returns:
        VIP layer index (0-based).
    """
    # Prefer discrimination signal when counterfactuals are available
    signal = d_disc if d_disc is not None else d_vis

    baseline = signal[min_layer:min_layer + baseline_layers]
    threshold = baseline.mean() + threshold_k * baseline.std()

    for i in range(min_layer, len(signal)):
        if signal[i] > threshold:
            return i

    # Fallback: return the layer with the steepest rise
    diffs = np.diff(signal[min_layer:])
    return int(np.argmax(diffs)) + min_layer


def aggregate_vip(
    results: list[dict],
    baseline_layers: int = 4,
    threshold_k: float = 2.0,
) -> dict:
    """Compute mean distance curves and aggregate VIP across a set of samples.

    Args:
        results: List of dicts, each from compute_layer_distances().
        baseline_layers: Passed to detect_vip.
        threshold_k: Passed to detect_vip.

    Returns:
        dict with:
            'mean_d_vis':  np.ndarray [n_layers]
            'mean_d_cf':   np.ndarray [n_layers] or None
            'mean_d_disc': np.ndarray [n_layers] or None
            'vip_per_sample': list[int]
            'vip_mean':    float
            'vip_median':  int
    """
    d_vis_stack = np.stack([r["d_vis"] for r in results], axis=0)   # (n, n_layers)
    mean_d_vis = d_vis_stack.mean(axis=0)

    has_cf = all(r["d_cf"] is not None for r in results)
    mean_d_cf = mean_d_disc = None
    if has_cf:
        mean_d_cf = np.stack([r["d_cf"] for r in results], axis=0).mean(axis=0)
        mean_d_disc = np.stack([r["d_disc"] for r in results], axis=0).mean(axis=0)

    vip_per_sample = [
        detect_vip(
            r["d_vis"],
            d_cf=r.get("d_cf"),
            d_disc=r.get("d_disc"),
            baseline_layers=baseline_layers,
            threshold_k=threshold_k,
        )
        for r in results
    ]

    return {
        "mean_d_vis": mean_d_vis,
        "mean_d_cf": mean_d_cf,
        "mean_d_disc": mean_d_disc,
        "vip_per_sample": vip_per_sample,
        "vip_mean": float(np.mean(vip_per_sample)),
        "vip_median": int(np.median(vip_per_sample)),
    }
