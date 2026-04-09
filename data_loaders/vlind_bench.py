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

Metrics (matching lmms-eval tasks/vlind_bench/utils.py):
    s_ck         — CK pass rate over all instances
    s_vp         — VP pass rate over all instances
    s_cb         — CB pass rate, denominator = CK-passing instances
    s_lp         — LP pass rate (macro), denominator = CK+VP+CB-passing instances
                   Per instance: fraction of good CF images where both LP q1 and q2 pass.
                   This is the **main metric** for language-prior measurement.
    accuracy_lp  — Raw LP item accuracy, no pipeline conditioning

``is_match`` uses left-to-right word scan (matching lmms-eval's
``infer_true_or_false``) to handle model outputs like "True, because..." or
"True." robustly.  Returns True iff the first "true"/"false" word found matches
the target (after lowercase).
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _extract_true_false(text: str) -> str | None:
    """Left-to-right word scan for 'true'/'false'.

    Ported from lmms-eval tasks/vlind_bench/utils.py ``infer_true_or_false``.
    Returns 'true', 'false', or None.
    """
    for word in text.lower().replace("\n", " ").replace(",", "").replace(".", "").split():
        if word == "true":
            return "true"
        if word == "false":
            return "false"
    return None


def is_match(pred: str, target: str) -> bool:
    """Match VLind-Bench prediction against target using word-scan extraction.

    Matches lmms-eval's infer_true_or_false logic: scans left-to-right for the
    first 'true'/'false' word in the prediction, then compares to the target
    (also normalised to lowercase).  Returns False if no true/false word found.
    """
    extracted = _extract_true_false(pred)
    if extracted is None:
        return False
    return extracted == target.strip().lower()


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


def load_vlind_bench_lp(n_samples: int | None = None) -> list[dict]:
    """Load VLind-Bench LP-stage items for evaluation experiments (REVIS, steering, EVA).

    Returns one sample per (instance × good_cf_image × qid) with CF images pre-loaded,
    using the standard eval sample interface.

    Unlike ``load_vlind_bench`` (which returns factual-image pairs for CoE), this
    function returns CF images with LP prompts — the correct setup for measuring
    whether an intervention improves language-prior resistance.

    Args:
        n_samples: Maximum number of *instances* to load.  ``None`` = all 302.

    Returns:
        List of sample dicts with keys:
            id          — "{instance_id}_lp_{cf_img_idx}_{qid}"
            image       — PIL Image (CF image for this LP item)
            cf_image    — None
            messages    — LP prompt (true_statement for q1, false_statement for q2)
            answer      — "true" (q1) or "false" (q2)
            instance_id — for compute_vlind_metrics() grouping
            stage       — "lp"
            qid         — "q1" or "q2"
            cf_img_idx  — CF image index (for per-image LP pass computation)
    """
    raw_data, factual_dir, cf_dir = _download_and_parse()
    if n_samples is not None:
        raw_data = raw_data[:n_samples]

    samples = []
    n_skipped_img = 0
    for idx, entry in enumerate(raw_data):
        concept = entry["concept"]
        context_id = entry["context_id"]

        good_ids = sorted(
            int(k)
            for k, v in entry.get("aggregated_human_label_good_images", {}).items()
            if isinstance(v, (int, float)) and v >= 2
        )
        if not good_ids:
            continue

        cf_paths = _find_cf_paths(cf_dir, concept, context_id)
        true_stmt = entry["true_statement"]
        false_stmt = entry["false_statement"]

        for cf_idx in good_ids:
            cf_path = cf_paths[cf_idx] if cf_idx < len(cf_paths) else ""
            cf_image = _open_image(cf_path)
            if cf_image is None:
                n_skipped_img += 1
                continue

            for qid, stmt, ans in [
                ("q1", true_stmt,  "true"),
                ("q2", false_stmt, "false"),
            ]:
                prompt = (
                    f"Statement: {stmt}\n"
                    "Based on the image, is the given statement true or false? "
                    "Forget real-world common sense and just follow the information "
                    "provided in the image.\n"
                    "Only respond in True or False."
                )
                samples.append({
                    "id":          f"{idx}_lp_{cf_idx}_{qid}",
                    "image":       cf_image,
                    "cf_image":    None,
                    "messages":    [{"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ]}],
                    "answer":      ans,
                    # pipeline fields for compute_vlind_metrics()
                    "instance_id": idx,
                    "stage":       "lp",
                    "qid":         qid,
                    "cf_img_idx":  cf_idx,
                })

    logger.info(
        "Loaded %d VLind-Bench LP eval samples (%d instances, %d CF images skipped).",
        len(samples), len(raw_data), n_skipped_img,
    )
    return samples


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------

def compute_vlind_metrics(results: list[dict]) -> dict:
    """Compute VLind-Bench pipeline metrics from per-item result dicts.

    Matches lmms-eval tasks/vlind_bench/utils.py aggregation exactly.

    Each result dict must have:
        instance_id  — int, groups items into instances
        stage        — "ck" | "vp" | "cb" | "lp"
        qid          — "q1" | "q2"
        cf_img_idx   — int (used for LP per-image grouping; -1 for non-LP)
        is_correct   — bool

    Returns:
        s_ck         — CK pass rate (all instances)
        s_vp         — VP pass rate (all instances)
        s_cb         — CB pass rate (CK-passing instances; None if none pass)
        s_lp         — LP pass rate macro (CK+VP+CB-passing; None if none qualify)
        accuracy_lp  — raw LP item accuracy (no conditioning)
        n_instances  — total number of instances
        n_lp_qualifying — number of instances qualifying for s_lp
    """
    # Group by instance
    groups: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        groups[r["instance_id"]].append(r)

    def _stage_passes(items: list[dict], stage: str) -> bool:
        stage_items = [r for r in items if r["stage"] == stage]
        q1_ok = any(r["is_correct"] for r in stage_items if r["qid"] == "q1")
        q2_ok = any(r["is_correct"] for r in stage_items if r["qid"] == "q2")
        return q1_ok and q2_ok

    ck_passes, vp_passes = [], []
    cb_num, cb_den = 0, 0
    lp_instance_scores: list[float] = []

    for items in groups.values():
        ck_ok = _stage_passes(items, "ck")
        vp_ok = _stage_passes(items, "vp")
        cb_ok = _stage_passes(items, "cb")
        ck_passes.append(ck_ok)
        vp_passes.append(vp_ok)

        if ck_ok:
            cb_den += 1
            if cb_ok:
                cb_num += 1

        if ck_ok and vp_ok and cb_ok:
            # LP: per good-image pass rate
            by_img: dict[int, list[dict]] = defaultdict(list)
            for r in items:
                if r["stage"] == "lp":
                    by_img[r["cf_img_idx"]].append(r)
            img_passes = [
                any(r["is_correct"] for r in img_items if r["qid"] == "q1")
                and any(r["is_correct"] for r in img_items if r["qid"] == "q2")
                for img_items in by_img.values()
            ]
            if img_passes:
                lp_instance_scores.append(sum(img_passes) / len(img_passes))

    # Raw LP accuracy
    lp_items = [r for r in results if r["stage"] == "lp"]
    accuracy_lp = (sum(r["is_correct"] for r in lp_items) / len(lp_items)
                   if lp_items else 0.0)

    n = len(groups)
    return {
        "s_ck":            sum(ck_passes) / n if n else 0.0,
        "s_vp":            sum(vp_passes) / n if n else 0.0,
        "s_cb":            cb_num / cb_den if cb_den > 0 else None,
        "s_lp":            (sum(lp_instance_scores) / len(lp_instance_scores)
                            if lp_instance_scores else None),
        "accuracy_lp":     accuracy_lp,
        "n_instances":     n,
        "n_lp_qualifying": len(lp_instance_scores),
    }
