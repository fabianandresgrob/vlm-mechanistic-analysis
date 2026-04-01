"""ViLP dataset loader.

HuggingFace repo: ``ViLP/ViLP``
Canonical split:  ``train``

Each item has three images (image1/image2/image3) and three answers (answer1/2/3).
For chain-of-embedding three-condition analysis:
    vis   = image1  (standard image aligned with language prior)
    cf    = image2  (first counterfactual probe image)
    answer = answer1 (ground truth for the vis condition)

The question is prefixed with "Please answer with one word: " following the
lmms-eval ViLP task default (``with_fact=False`` mode; the fact prefix is omitted
because lmms-eval's default prompt already handles single-word forcing).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def is_match(pred: str, target: str) -> bool:
    """Exact match after lowercase + strip (ViLP answers are single words)."""
    return pred.strip().lower() == target.strip().lower()


def load_vilp(n_samples: int | None = None) -> list[dict]:
    """Load ViLP samples.

    Args:
        n_samples: Maximum number of samples.  ``None`` = all.

    Returns:
        List of sample dicts with keys: id, image, cf_image, messages, answer.
    """
    from datasets import load_dataset  # HuggingFace datasets

    logger.info("Loading ViLP (n=%s)…", n_samples or "all")
    ds = load_dataset("ViLP/ViLP", split="train")
    if n_samples is not None:
        ds = ds.select(range(min(n_samples, len(ds))))

    samples = []
    for item in ds:
        question = item.get("question", "")
        prompt = f"Please answer with one word: {question}"
        image1 = item.get("image1")
        image2 = item.get("image2")
        samples.append(
            {
                "id": item.get("id", len(samples)),
                "image": image1.convert("RGB") if image1 else None,
                "cf_image": image2.convert("RGB") if image2 else None,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                "answer": str(item.get("answer1", "")).strip().lower(),
            }
        )

    logger.info(
        "Loaded %d ViLP samples (%d with cf images).",
        len(samples),
        sum(s["cf_image"] is not None for s in samples),
    )
    return samples
