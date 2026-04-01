"""VLMs Are Biased (VAB) dataset loader.

HuggingFace repo: ``anvo25/vlms-are-biased``
Canonical split:  ``main`` (contains ~720 rows = 240 unique questions × 3 resolutions)

Field mapping (from HF dataset):
    ID             → id
    image          → image
    prompt         → messages[0].content[1].text
    ground_truth   → answer  (normalized: strip {}, lowercase)
    expected_bias  → expected_bias  (normalized)
    topic          → topic
    sub_topic      → sub_topic
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _normalize(s: str) -> str:
    return s.strip().strip("{}").strip().lower()


def is_match(pred: str, target: str) -> bool:
    """Match prediction against target following lmms-eval VAB logic.

    1. Exact match after normalization (strip {}, lowercase).
    2. Numeric fallback: extract digits and compare (handles "2" vs "{2}").
    """
    pred_n = _normalize(pred)
    tgt_n = _normalize(target)
    if pred_n == tgt_n:
        return True
    pred_digits = "".join(c for c in pred_n if c.isdigit())
    tgt_digits = "".join(c for c in tgt_n if c.isdigit())
    return bool(pred_digits) and bool(tgt_digits) and pred_digits == tgt_digits


def load_vab(
    dataset_id: str = "anvo25/vlms-are-biased",
    split: str = "main",
    n_samples: int | None = None,
) -> list[dict]:
    """Load VAB samples.

    Args:
        dataset_id: HuggingFace dataset repo ID.
        split:      Dataset split — use ``"main"`` (not ``"test"``).
        n_samples:  Maximum number of samples to return.  ``None`` = all.

    Returns:
        List of sample dicts with keys: id, image, cf_image, messages, answer,
        expected_bias, topic, sub_topic.
    """
    from datasets import load_dataset  # HuggingFace datasets

    logger.info("Loading VAB from %s / %s…", dataset_id, split)
    ds = load_dataset(dataset_id, split=split)
    if n_samples is not None:
        ds = ds.select(range(min(n_samples, len(ds))))

    samples = []
    for item in ds:
        samples.append(
            {
                "id": item.get("ID") or len(samples),
                "image": item.get("image"),
                "cf_image": None,  # VAB has no paired counterfactual images
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": item.get("prompt") or ""},
                        ],
                    }
                ],
                "answer": _normalize(item.get("ground_truth") or ""),
                "expected_bias": _normalize(item.get("expected_bias") or ""),
                "topic": item.get("topic", ""),
                "sub_topic": item.get("sub_topic", ""),
            }
        )

    logger.info("Loaded %d VAB samples.", len(samples))
    return samples
