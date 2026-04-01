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

from .vab import load_vab
from .vilp import load_vilp
from .vlind_bench import expand_vlind_bench_stages, load_vlind_bench
from .vqav2 import load_vqav2


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
    "load_vilp",
    "load_vlind_bench",
    "expand_vlind_bench_stages",
    "load_vqav2",
    "to_contrastive_sample",
]
