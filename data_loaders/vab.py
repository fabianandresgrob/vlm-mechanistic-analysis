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
    resolution: int | None = 1152,
) -> list[dict]:
    """Load VAB samples.

    The ``main`` split contains each (image, question) pair at 3 resolutions
    (384, 768, 1152 px).  ``resolution`` filters to a single resolution to avoid
    triple-counting concepts.  Use ``resolution=None`` to load all rows (matches
    raw lmms-eval behaviour).

    Args:
        dataset_id:  HuggingFace dataset repo ID.
        split:       Dataset split — use ``"main"`` (not ``"test"``).
        n_samples:   Maximum number of samples to return after filtering.
                     ``None`` = all.
        resolution:  Keep only rows whose image filename ends in this resolution
                     (e.g. 1152 → keeps ``*_1152.png`` / ``*_px1152.png``).
                     Default is 1152 (highest resolution, per supervisor guidance).
                     Pass ``None`` to disable filtering.

    Returns:
        List of sample dicts with keys: id, image, cf_image, messages, answer,
        expected_bias, topic, sub_topic.
    """
    import re

    from datasets import load_dataset  # HuggingFace datasets

    logger.info("Loading VAB from %s / %s…", dataset_id, split)
    ds = load_dataset(dataset_id, split=split)
    if n_samples is not None:
        ds = ds.select(range(min(n_samples, len(ds))))

    _res_pattern = re.compile(r"(?:_px|_)(\d+)\.(?:png|jpg)$", re.IGNORECASE)

    samples = []
    for item in ds:
        if resolution is not None:
            img_name = ""
            # Try to get the filename from the image object
            raw_img = item.get("image")
            if hasattr(raw_img, "filename") and raw_img.filename:
                img_name = raw_img.filename
            # Fall back to the image_path field if present
            if not img_name:
                img_name = str(item.get("image_path") or "")
            m = _res_pattern.search(img_name)
            if m and int(m.group(1)) != resolution:
                continue

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


def load_vab_pairs(
    dataset_id: str = "fabiangrob/vlms-bias-pairs",
    split: str = "train",
    n_samples: int | None = None,
) -> list[dict]:
    """Load VAB counterfactual pairs from ``fabiangrob/vlms-bias-pairs``.

    Each sample has a paired original + modified image (subtle visual changes,
    e.g. added leg, optical illusions) suitable for the full three-condition
    chain-of-embedding analysis.

    Field mapping:
        id             → id
        original_image → image   (condition B — original)
        modified_image → cf_image (condition C — counterfactual)
        prompt         → messages[0].content[1].text
        ground_truth   → answer  (normalized: strip {}, lowercase)
        expected_bias  → expected_bias
        topic          → topic
        sub_topic      → sub_topic

    Args:
        dataset_id: HuggingFace dataset repo ID.
        split:      Dataset split (default: ``"train"``).
        n_samples:  Maximum number of samples to return.  ``None`` = all.

    Returns:
        List of sample dicts with keys: id, image, cf_image, messages, answer,
        expected_bias, topic, sub_topic.
    """
    from datasets import load_dataset  # HuggingFace datasets

    logger.info("Loading VAB pairs from %s / %s…", dataset_id, split)
    ds = load_dataset(dataset_id, split=split)
    if n_samples is not None:
        ds = ds.select(range(min(n_samples, len(ds))))

    samples = []
    for item in ds:
        samples.append(
            {
                "id": item.get("id") or len(samples),
                "image": item.get("original_image"),
                "cf_image": item.get("modified_image"),
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

    n_with_cf = sum(s["cf_image"] is not None for s in samples)
    logger.info("Loaded %d VAB pair samples (%d with cf images).", len(samples), n_with_cf)
    return samples
