"""VQAv2 dataset loader.

HuggingFace repo: ``lmms-lab/VQAv2``
Canonical split:  ``validation``
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def load_vqav2(
    n_samples: int | None = None,
    split: str = "validation",
) -> list[dict]:
    """Load VQAv2 samples.

    Args:
        n_samples: Maximum number of samples.  ``None`` = all.
        split:     Dataset split (default ``"validation"``).

    Returns:
        List of sample dicts with keys: id, image, cf_image, messages, answer.
        ``cf_image`` is always ``None`` (VQAv2 has no counterfactuals).
    """
    from datasets import load_dataset  # HuggingFace datasets

    logger.info("Loading VQAv2 %s (n=%s)…", split, n_samples or "all")
    ds = load_dataset("lmms-lab/VQAv2", split=split)
    if n_samples is not None:
        ds = ds.select(range(min(n_samples, len(ds))))

    samples = []
    for item in ds:
        samples.append(
            {
                "id": item.get("question_id", len(samples)),
                "image": item["image"],
                "cf_image": None,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": item["question"]},
                        ],
                    }
                ],
                "answer": item.get("multiple_choice_answer", ""),
            }
        )

    logger.info("Loaded %d VQAv2 samples.", len(samples))
    return samples
