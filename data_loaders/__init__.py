"""Centralized dataset loaders for VLM mechanistic analysis.

All loaders return ``list[dict]`` with a consistent interface::

    {
        "id":            any        — unique sample identifier
        "image":         PIL.Image  — primary visual input (None for blind condition)
        "cf_image":      PIL.Image  — counterfactual image (None if not applicable)
        "messages":      list[dict] — chat-template messages (role/content format)
        "answer":        str        — ground-truth answer, normalized to lowercase
        # dataset-specific extras (present only where applicable):
        "expected_bias": str        — VAB biased answer
        "topic":         str        — VAB topic
        "sub_topic":     str        — VAB sub-topic
        "concept":       str        — VLind-Bench concept
        "context":       str        — VLind-Bench context
    }

Usage
-----
    from data_loaders import load_vab, load_vilp, load_vlind_bench, load_vqav2

    # For chain-of-embedding, convert to ContrastiveSample:
    from data_loaders import to_contrastive_sample
    cs_samples = [to_contrastive_sample(d) for d in load_vab()]

Note: this package is named ``data_loaders`` (not ``datasets``) to avoid shadowing
the HuggingFace ``datasets`` library.  Import as shown above.
"""

from .vab import is_match as vab_is_match
from .vab import load_vab, load_vab_pairs
from .vilp import compute_vilp_metrics
from .vilp import is_match as vilp_is_match
from .vilp import load_vilp, load_vilp_expanded
from .vilp import normalize_output as vilp_normalize_output
from .vlind_bench import (compute_vlind_metrics, expand_vlind_bench_stages,
                          load_vlind_bench, load_vlind_bench_lp)
from .vlind_bench import is_match as vlind_is_match
from .vqav2 import load_vqav2


def get_is_match(dataset: str):
    """Return the correct is_match function for the given dataset name."""
    return {
        "vlms_are_biased": vab_is_match,
        "vab": vab_is_match,
        "vab_pairs": vab_is_match,
        "vilp": vilp_is_match,
        "vlind": vlind_is_match,
        "vqav2": lambda p, t: p.strip().lower() == t.strip().lower(),
    }.get(dataset, lambda p, t: p.strip().lower() == t.strip().lower())


def to_contrastive_sample(d: dict):
    """Convert a standard sample dict to ContrastiveSample for chain-of-embedding."""
    from chain_of_embedding.contrastive_forward import ContrastiveSample

    return ContrastiveSample(
        id=d["id"],
        image=d.get("image"),
        cf_image=d.get("cf_image"),
        messages=d["messages"],
        answer=d.get("answer", ""),
    )


__all__ = [
    "load_vab",
    "load_vab_pairs",
    "load_vilp",
    "load_vilp_expanded",
    "load_vlind_bench",
    "load_vlind_bench_lp",
    "expand_vlind_bench_stages",
    "load_vqav2",
    "vab_is_match",
    "vilp_is_match",
    "vilp_normalize_output",
    "compute_vilp_metrics",
    "vlind_is_match",
    "compute_vlind_metrics",
    "get_is_match",
    "to_contrastive_sample",
]
