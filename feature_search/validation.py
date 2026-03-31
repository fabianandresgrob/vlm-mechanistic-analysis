"""
Exp 3.2: Statistical Validation on VLMs Are Biased.

Two analyses:
  A) Layer-level: JS divergence / cosine distance between original and
     counterfactual conditions (B vs C). Test whether correctly-answered
     samples show higher divergence than biased samples.

  B) Feature-level: Do the top visual reliance latents from Exp 3.1 show
     differential activation between original vs counterfactual conditions
     on VAB? Tests with Mann-Whitney U (non-parametric, no distributional
     assumption) and reports effect size (rank-biserial correlation).
"""
from __future__ import annotations
import logging
import numpy as np
from scipy.stats import mannwhitneyu

logger = logging.getLogger(__name__)


def rank_biserial_correlation(u_stat: float, n1: int, n2: int) -> float:
    """Effect size for Mann-Whitney U.

    scipy.stats.mannwhitneyu returns U1 = number of (x_i, y_j) pairs where x_i > y_j.
    The standard rank-biserial r is +1 when x >> y, so:
        r = 2 * U1 / (n1 * n2) - 1

    Range [-1, 1]: +1 means x (scores_correct) >> y (scores_biased).
    """
    return float((2 * u_stat) / (n1 * n2) - 1.0)


def test_condition_divergence(
    scores_correct: np.ndarray,
    scores_biased: np.ndarray,
    alternative: str = "greater",
) -> dict:
    """Mann-Whitney U test: are divergence scores higher for correct samples?

    Args:
        scores_correct: Per-sample divergence/distance scores for correctly-
                        answered samples. Shape (n_correct,).
        scores_biased:  Per-sample divergence/distance scores for biased
                        (incorrectly-answered) samples. Shape (n_biased,).
        alternative:    'greater' tests H1: correct > biased.

    Returns:
        dict with u_stat, pvalue, effect_size_r, n_correct, n_biased,
        mean_correct, mean_biased.
    """
    if len(scores_correct) < 5 or len(scores_biased) < 5:
        logger.warning("Too few samples for reliable test (n_correct=%d, n_biased=%d).",
                       len(scores_correct), len(scores_biased))

    result = mannwhitneyu(scores_correct, scores_biased, alternative=alternative)
    effect_size = rank_biserial_correlation(
        result.statistic, len(scores_correct), len(scores_biased)
    )

    return {
        "u_stat": float(result.statistic),
        "pvalue": float(result.pvalue),
        "effect_size_r": effect_size,
        "n_correct": int(len(scores_correct)),
        "n_biased": int(len(scores_biased)),
        "mean_correct": float(np.mean(scores_correct)),
        "mean_biased": float(np.mean(scores_biased)),
    }


test_condition_divergence.__test__ = False  # prevent pytest from collecting this as a test


def feature_activation_test(
    feat_acts_condition_b: np.ndarray,
    feat_acts_condition_c: np.ndarray | None,
    latent_indices: list[int],
    is_correct: np.ndarray,
) -> list[dict]:
    """For each latent in latent_indices, test differential activation on VAB.

    Tests whether the latent activation magnitude differs between:
      - Samples where model answers correctly (uses vision)
      - Samples where model gives biased answer (ignores vision)

    Args:
        feat_acts_condition_b: SAE activations for condition B (original).
                                Shape (n_samples, d_sae).
        feat_acts_condition_c: SAE activations for condition C (counterfactual).
                                Shape (n_samples, d_sae). Can be None if no CF.
        latent_indices:        List of latent indices to test.
        is_correct:            Binary array (n_samples,) — 1 if model correct.

    Returns:
        List of dicts (one per latent) with test results, sorted by effect_size_r desc.
    """
    correct_mask = is_correct.astype(bool)
    biased_mask = ~correct_mask
    results = []

    for j in latent_indices:
        acts_j = feat_acts_condition_b[:, j]  # (n_samples,)
        test = test_condition_divergence(
            acts_j[correct_mask],
            acts_j[biased_mask],
        )
        test["latent_idx"] = j
        results.append(test)

    results.sort(key=lambda x: x["effect_size_r"], reverse=True)
    return results
