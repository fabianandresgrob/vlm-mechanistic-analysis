"""Shared utilities for vlind-bench-oe evaluation.

Scoring logic is shared between run_vlind_bench_oe.py and the interactive notebook.
The lmms-eval task duplicates this to avoid cross-repo imports.
"""

from __future__ import annotations

import re


def _normalize(text: str) -> str:
    """Lowercase, strip articles and punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)          # punctuation → space
    text = re.sub(r"\b(a|an|the)\b", " ", text)   # remove articles
    return " ".join(text.split())                  # collapse whitespace


def score_response(
    response: str,
    expected_answers: list[str],
    biased_answers: list[str],
) -> str:
    """Classify a model response as 'correct', 'biased', or 'other'.

    Matching is substring-based after normalization (lowercase, no articles,
    no punctuation).  Both directions are checked:
      - model answer contained in a reference answer  ("desert" in "desert sands")
      - reference answer contained in model answer    ("desert" in "sandy desert area")

    Expected answers are checked before biased answers — if the response matches
    both (unlikely but possible), it is labelled correct.
    """
    resp = _normalize(response)
    if not resp:
        return "other"

    for ans in expected_answers:
        a = _normalize(ans)
        if a and (a in resp or resp in a):
            return "correct"

    for ans in biased_answers:
        a = _normalize(ans)
        if a and (a in resp or resp in a):
            return "biased"

    return "other"


def format_prompt(question: str, instructions: str) -> str:
    """Combine question and instructions into the full model prompt."""
    return question + instructions
