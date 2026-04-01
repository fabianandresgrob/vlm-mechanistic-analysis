"""VLind-Bench dataset loader.

Downloads the raw imagefolder repo from ``klee972/VLind-Bench`` via
``snapshot_download`` (cached in HF_HOME after the first ~5 GB download) and
builds samples in-memory — mirroring the approach used in the lmms-eval
``VLindBenchTask``.

The dataset contains 302 instances, each with four pipeline stages:

    CK  — Conceptual Knowledge  (factual image)
    VP  — Visual Perception     (best CF image)
    CB  — Context Binding       (best CF image + context)
    LP  — Language Prior        (CF image, no context hint)

``load_vlind_bench`` returns one sample per instance paired for
chain-of-embedding analysis (vis = CK factual, cf = LP best-CF).

``expand_vlind_bench_stages`` returns the full per-stage expansion
(mirrors lmms-eval's _expand_all) for comprehensive evaluation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def is_match(pred: str, target: str) -> bool:
    """Exact match after lowercase + strip (VLind-Bench answers are 'true'/'false')."""
    return pred.strip().lower() == target.strip().lower()


# ---------------------------------------------------------------------------
# Internal path helpers
# ---------------------------------------------------------------------------

def _find_factual_path(factual_dir: Path, concept: str, context_id: int) -> str:
    d = factual_dir / concept
    if d.exists():
        prefix = f"{context_id}_"
        for folder in d.iterdir():
            if folder.name.startswith(prefix):
                p = folder / "0.jpg"
                if p.exists():
                    return str(p)
    return ""


def _find_cf_paths(cf_dir: Path, concept: str, context_id: int) -> list[str]:
    """Return up to 12 CF image paths (empty string where missing)."""
    d = cf_dir / concept
    if d.exists():
        prefix = f"{context_id}_"
        for folder in d.iterdir():
            if folder.name.startswith(prefix):
                return [
                    str(p) if (p := folder / f"{i}.jpg").exists() else ""
                    for i in range(12)
                ]
    return [""] * 12


def _open_image(path: str):
    from PIL import Image

    if path and Path(path).exists():
        return Image.open(path).convert("RGB")
    return None


def _lp_prompt(true_statement: str) -> str:
    return (
        f"Statement: {true_statement}\n"
        "Based on the image, is the given statement true or false? "
        "Forget real-world common sense and just follow the information "
        "provided in the image.\n"
        "Only respond in True or False."
    )


def _download_and_parse() -> tuple[list[dict], Path, Path]:
    """Download VLind-Bench repo and return (raw_data, factual_dir, cf_dir)."""
    from huggingface_hub import snapshot_download

    logger.info(
        "Downloading/loading VLind-Bench from HuggingFace "
        "(cached in HF_HOME after the first ~5 GB download)…"
    )
    local_dir = Path(snapshot_download(repo_id="klee972/VLind-Bench", repo_type="dataset"))

    data_root = local_dir / "VLind-Bench Dataset"
    if not data_root.exists():
        data_root = local_dir

    with open(data_root / "data.json") as f:
        raw_data = json.load(f)

    factual_dir = data_root / "images" / "factual"
    cf_dir = data_root / "images" / "counterfactual"
    return raw_data, factual_dir, cf_dir


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_vlind_bench(n_samples: int | None = None) -> list[dict]:
    """Load VLind-Bench samples for VIP/TVI analysis.

    Each returned sample pairs:
        image    = CK factual image (vis condition)
        cf_image = best LP counterfactual image (cf condition)
        messages = LP q1 prompt (true_statement, answer = "true")
        answer   = "true"

    Args:
        n_samples: Maximum number of *instances* to load.  ``None`` = all 302.

    Returns:
        List of sample dicts with keys: id, image, cf_image, messages, answer,
        concept, context, true_statement, false_statement, factual_path, best_cf_path.
    """
    raw_data, factual_dir, cf_dir = _download_and_parse()
    if n_samples is not None:
        raw_data = raw_data[:n_samples]

    samples = []
    n_skipped = 0
    for idx, entry in enumerate(raw_data):
        concept = entry["concept"]
        context_id = entry["context_id"]
        best = int(entry.get("best_img_id", 0))

        factual_path = _find_factual_path(factual_dir, concept, context_id)
        cf_paths = _find_cf_paths(cf_dir, concept, context_id)
        best_cf_path = cf_paths[best] if best < len(cf_paths) else ""

        vis_image = _open_image(factual_path)
        if vis_image is None:
            n_skipped += 1
            continue

        cf_image = _open_image(best_cf_path)
        true_stmt = entry["true_statement"]

        samples.append(
            {
                "id": idx,
                "image": vis_image,
                "cf_image": cf_image,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": _lp_prompt(true_stmt)},
                        ],
                    }
                ],
                "answer": "true",
                # metadata
                "concept": concept,
                "context": entry.get("context", ""),
                "true_statement": true_stmt,
                "false_statement": entry["false_statement"],
                "factual_path": factual_path,
                "best_cf_path": best_cf_path,
            }
        )

    logger.info(
        "Loaded %d VLind-Bench samples (%d skipped, %d with cf images).",
        len(samples),
        n_skipped,
        sum(s["cf_image"] is not None for s in samples),
    )
    return samples


def expand_vlind_bench_stages(n_samples: int | None = None) -> list[dict[str, Any]]:
    """Expand VLind-Bench into all four pipeline stages (CK/VP/CB/LP).

    Returns a list of per-stage evaluation items — one per (instance × stage × qid)
    combination, mirroring lmms-eval's ``VLindBenchTask._expand_all``.  Images are
    **not** pre-loaded; use ``_image_path`` and open with PIL as needed.

    Each item includes:
        _stage, _qid, _stmt_key / _noun_key, _image_path, answer ("true"/"false"),
        plus all base fields (instance_id, concept, context, true_statement, …).

    Args:
        n_samples: Maximum number of *instances* to expand.  ``None`` = all 302.
    """
    raw_data, factual_dir, cf_dir = _download_and_parse()
    if n_samples is not None:
        raw_data = raw_data[:n_samples]

    expanded: list[dict[str, Any]] = []

    for idx, entry in enumerate(raw_data):
        concept = entry["concept"]
        context_id = entry["context_id"]
        best = int(entry.get("best_img_id", 0))

        good_ids = sorted(
            int(k)
            for k, v in entry.get("aggregated_human_label_good_images", {}).items()
            if isinstance(v, (int, float)) and v >= 2
        )

        factual_path = _find_factual_path(factual_dir, concept, context_id)
        cf_paths = _find_cf_paths(cf_dir, concept, context_id)
        best_cf_path = cf_paths[best] if best < len(cf_paths) else ""

        base: dict[str, Any] = {
            "instance_id": idx,
            "concept": concept,
            "context": entry["context"],
            "true_statement": entry["true_statement"],
            "false_statement": entry["false_statement"],
            "existent_noun": entry.get("existent_noun", ""),
            "non_existent_noun": entry.get(
                "non-existent_noun", entry.get("non_existent_noun", "")
            ),
        }

        # CK — factual image
        for qid, stmt_key, ans in [("q1", "false_statement", "true"),
                                    ("q2", "true_statement", "false")]:
            expanded.append(
                {**base, "_stage": "ck", "_qid": qid, "_stmt_key": stmt_key,
                 "_image_path": factual_path, "answer": ans}
            )

        # VP — best CF image
        for qid, noun_key, ans in [("q1", "existent_noun", "true"),
                                    ("q2", "non_existent_noun", "false")]:
            expanded.append(
                {**base, "_stage": "vp", "_qid": qid, "_noun_key": noun_key,
                 "_image_path": best_cf_path, "answer": ans}
            )

        # CB — best CF image
        for qid, stmt_key, ans in [("q1", "true_statement", "true"),
                                    ("q2", "false_statement", "false")]:
            expanded.append(
                {**base, "_stage": "cb", "_qid": qid, "_stmt_key": stmt_key,
                 "_image_path": best_cf_path, "answer": ans}
            )

        # LP — one item per good CF image
        for cf_idx in good_ids:
            cf_path = cf_paths[cf_idx] if cf_idx < len(cf_paths) else ""
            for qid, stmt_key, ans in [("q1", "true_statement", "true"),
                                        ("q2", "false_statement", "false")]:
                expanded.append(
                    {**base, "_stage": "lp", "_qid": qid, "_stmt_key": stmt_key,
                     "_cf_img_idx": cf_idx, "_image_path": cf_path, "answer": ans}
                )

    logger.info("Expanded %d VLind-Bench stage items from %d instances.", len(expanded), len(raw_data))
    return expanded
