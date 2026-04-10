"""vlind-bench-oe dataset loader.

Loads ``fabiangrob/vlind-bench-oe`` from HuggingFace and returns samples
in the same interface used throughout this repo (image / cf_image / messages /
answer), so all experiments (CoE, EVA, VTI, steering) can consume this dataset
without special-casing.

Dataset schema (one row = one prompt-ready datapoint):
    instance_id      int     — links rows from the same source instance
    cf_img_idx       int     — which of the ≤12 CF images
    concept          str     — habitat / diet / color / size / …
    question         str     — open-ended question
    instructions     str     — instruction suffix (e.g. "Answer in one to three words.")
    expected_answers list    — answers consistent with the CF image
    biased_answers   list    — answers the language prior would predict
    true_statement   str     — original VLind-Bench true_statement (debug)
    false_statement  str     — original VLind-Bench false_statement (debug)
    cf_image         Image   — counterfactual PIL image
    factual_image    Image   — original real-world PIL image

Sample interface (mirrors data_loaders/vlind_bench.py):
    id               str     — "{instance_id}_{cf_img_idx}"
    image            PIL     — factual image  (vis condition for CoE)
    cf_image         PIL     — CF image       (cf condition for CoE)
    messages         list    — chat-format prompt with cf_image
    answer           str     — expected_answers[0]  (primary target for display)
    expected_answers list    — all correct answers (for score_sample)
    biased_answers   list    — all biased answers   (for score_sample)
    instance_id      int
    cf_img_idx       int
    concept          str
    question         str
    instructions     str
    true_statement   str
    false_statement  str

Scoring:
    score_sample(pred, sample) → 'correct' | 'biased' | 'other'
    compute_oe_metrics(results) → dict with accuracy, bias_rate, other_rate,
                                  per_concept breakdown, n_total
"""

from __future__ import annotations

import logging
import sys
import os
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_creation.vlind_bench_oe_utils import score_response

logger = logging.getLogger(__name__)

DEFAULT_DATASET = "fabiangrob/vlind-bench-oe"
DEFAULT_INSTRUCTIONS = "\nAnswer the question using a single word or phrase."


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_vlind_bench_oe(
    n_samples: int | None = None,
    instructions: str | None = None,
    dataset_name: str = DEFAULT_DATASET,
) -> list[dict[str, Any]]:
    """Load vlind-bench-oe samples from HuggingFace.

    Each sample pairs the factual image (vis condition) with the CF image
    (cf condition) and an open-ended prompt, ready for CoE / EVA / VTI /
    steering experiments.

    Args:
        n_samples:    Maximum number of rows to load.  ``None`` = all.
        instructions: Override the instruction suffix stored in the dataset.
                      Pass ``None`` to use the dataset's stored instructions.
        dataset_name: HuggingFace dataset repo (default: fabiangrob/vlind-bench-oe).

    Returns:
        List of sample dicts (see module docstring for keys).
    """
    from datasets import load_dataset

    logger.info("Loading %s …", dataset_name)
    ds = load_dataset(dataset_name, split="train")
    if n_samples is not None:
        ds = ds.select(range(min(n_samples, len(ds))))
    logger.info("Loaded %d rows.", len(ds))

    samples = []
    n_missing_cf = 0
    n_missing_factual = 0

    for row in ds:
        cf_img = row["cf_image"]
        factual_img = row.get("factual_image")

        if cf_img is None:
            n_missing_cf += 1
            continue
        if factual_img is None:
            n_missing_factual += 1

        instr = instructions if instructions is not None else row["instructions"]
        prompt = row["question"] + instr

        samples.append({
            "id":               f"{row['instance_id']}_{row['cf_img_idx']}",
            # image interface — cf_image is the primary (what the model sees for LP)
            # factual_image is kept as 'image' for the vis condition in CoE
            "image":            factual_img,
            "cf_image":         cf_img,
            # chat-format prompt (uses cf_image — the LP / open-ended condition)
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            # answer fields
            "answer":           row["expected_answers"][0] if row["expected_answers"] else "",
            "expected_answers": row["expected_answers"],
            "biased_answers":   row["biased_answers"],
            # metadata
            "instance_id":      row["instance_id"],
            "cf_img_idx":       row["cf_img_idx"],
            "concept":          row["concept"],
            "question":         row["question"],
            "instructions":     instr,
            "true_statement":   row["true_statement"],
            "false_statement":  row["false_statement"],
        })

    if n_missing_cf:
        logger.warning("%d rows skipped (missing cf_image).", n_missing_cf)
    if n_missing_factual:
        logger.warning("%d rows have no factual_image (dataset may need add_factual_images.py).", n_missing_factual)

    logger.info(
        "Loaded %d vlind-bench-oe samples (%d with factual image).",
        len(samples),
        sum(s["image"] is not None for s in samples),
    )
    return samples


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_sample(pred: str, sample: dict[str, Any]) -> str:
    """Score a model prediction against a sample.

    Returns 'correct', 'biased', or 'other'.
    Delegates to vlind_bench_oe_utils.score_response (substring matching after
    normalization).
    """
    return score_response(pred, sample["expected_answers"], sample["biased_answers"])


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------

def compute_oe_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-sample result dicts into benchmark metrics.

    Each result dict must have:
        score    — 'correct' | 'biased' | 'other'
        concept  — str  (for per-concept breakdown)

    Returns:
        n_total      — int
        accuracy     — fraction correct
        bias_rate    — fraction biased
        other_rate   — fraction other
        per_concept  — dict[concept → {n, accuracy, bias_rate}]
    """
    n = len(results)
    if n == 0:
        return {"n_total": 0, "accuracy": 0.0, "bias_rate": 0.0, "other_rate": 0.0, "per_concept": {}}

    n_correct = sum(r["score"] == "correct" for r in results)
    n_biased  = sum(r["score"] == "biased"  for r in results)
    n_other   = sum(r["score"] == "other"   for r in results)

    by_concept: dict[str, list[str]] = {}
    for r in results:
        by_concept.setdefault(r["concept"], []).append(r["score"])

    per_concept = {
        c: {
            "n":         len(scores),
            "accuracy":  sum(s == "correct" for s in scores) / len(scores),
            "bias_rate": sum(s == "biased"  for s in scores) / len(scores),
        }
        for c, scores in sorted(by_concept.items())
    }

    return {
        "n_total":    n,
        "accuracy":   n_correct / n,
        "bias_rate":  n_biased  / n,
        "other_rate": n_other   / n,
        "per_concept": per_concept,
    }
