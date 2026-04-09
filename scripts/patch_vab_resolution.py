"""
Filter existing VAB result files to keep only 1152px resolution samples.

All VAB result files were produced when load_vab() loaded all 3 resolutions
(384, 768, 1152 px), triple-counting each concept.  This script filters every
JSONL to the 1152px rows only and rewrites the associated summary JSON files
from scratch.

Files patched (all under results/):
  revis/vab/*.jsonl                + revis summary
  steering/*/vab/**/*.jsonl        + sweep_summary.json + all_summaries.json
  eva_decoding/*/vab/**/*.jsonl    + summary.json

Run once on any machine that has the results directory:
    python scripts/patch_vab_resolution.py
    python scripts/patch_vab_resolution.py --results_dir /path/to/results
    python scripts/patch_vab_resolution.py --dry_run      # preview only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaders.vab import is_match as vab_is_match

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def _has_res(s: str, res: int) -> bool:
    """True iff ``res`` appears as a resolution token in the sample ID string.

    Looks for ``px<res>`` or ``_<res>`` followed by ``_``, ``.``, or end-of-string.
    Handles IDs like:
      VerticalHorizontal_001_Q1_notitle_px1152_Q1
      horse_2_2_1152
      Flag_of_Uruguay_stripes_8_1152_1
    """
    return bool(re.search(rf"(?:px|_){res}(?:_|\.|$)", str(s), re.IGNORECASE))


def _is_1152(sample_id) -> bool:
    """Keep this sample: it's 1152px, OR has no resolution token at all
    (single-resolution topics like Logos/Flags that don't encode resolution)."""
    s = str(sample_id)
    if _has_res(s, 1152):
        return True
    # If neither 384 nor 768 appears → single-resolution sample → keep
    return not _has_res(s, 384) and not _has_res(s, 768)


# ---------------------------------------------------------------------------
# REVIS helpers
# ---------------------------------------------------------------------------

def _revis_summary(records: list[dict], alpha: float) -> dict:
    n = len(records)
    if n == 0:
        return {}
    van = sum(r.get("is_correct_vanilla", False) for r in records) / n
    ste = sum(r.get("is_correct_steered", False) for r in records) / n
    return {"alpha": alpha, "vanilla_accuracy": van, "steered_accuracy": ste,
            "delta": ste - van, "n": n}


def patch_revis_vab(revis_dir: Path, dry_run: bool) -> None:
    vab_dir = revis_dir / "vab"
    if not vab_dir.exists():
        logger.info("No revis/vab directory found, skipping.")
        return

    jsonl_files = sorted(vab_dir.glob("*.jsonl"))
    summaries = []

    for path in jsonl_files:
        if path.suffix != ".jsonl" or path.stem.endswith(".bak"):
            continue

        records = [json.loads(l) for l in open(path)]
        filtered = [r for r in records if _is_1152(r.get("sample_id") or r.get("id", ""))]

        logger.info("REVIS %s: %d → %d samples", path.name, len(records), len(filtered))

        # Extract alpha from filename: revis_layer16_alpha-200.jsonl → -200
        m = re.search(r"alpha(-?\d+)", path.stem)
        alpha = float(m.group(1)) if m else 0.0

        summaries.append((alpha, filtered))

        if not dry_run:
            bak = path.with_suffix(".jsonl.bak")
            if not bak.exists():
                path.rename(bak)
            with open(path, "w") as f:
                for r in filtered:
                    f.write(json.dumps(r) + "\n")

    # Rewrite summary
    if summaries and not dry_run:
        summary_data = [_revis_summary(recs, alpha) for alpha, recs in summaries]
        summary_path = revis_dir / "vab_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary_data, f, indent=2)
        logger.info("Wrote %s", summary_path)


# ---------------------------------------------------------------------------
# Steering helpers
# ---------------------------------------------------------------------------

def _steering_summary(records: list[dict], alpha: float, latent_idx: int | None) -> dict:
    n = len(records)
    if n == 0:
        return {}
    van = sum(r.get("is_correct_vanilla", False) for r in records) / n
    ste = sum(r.get("is_correct_steered", False) for r in records) / n
    entry = {"alpha": alpha, "vanilla_accuracy": van, "steered_accuracy": ste,
             "delta": ste - van, "n": n}
    if latent_idx is not None:
        entry["latent_idx"] = latent_idx
    return entry


def patch_steering_vab(steering_root: Path, dry_run: bool) -> None:
    for model_dir in steering_root.iterdir():
        if not model_dir.is_dir():
            continue
        vab_dir = model_dir / "vab"
        if not vab_dir.exists():
            continue

        for layer_dir in vab_dir.iterdir():
            if not layer_dir.is_dir():
                continue

            all_summaries: list[dict] = []

            for latent_dir in sorted(layer_dir.iterdir()):
                if not latent_dir.is_dir():
                    continue
                # Extract latent_idx from directory name: latent_12002 → 12002
                m = re.match(r"latent_(\d+)", latent_dir.name)
                latent_idx = int(m.group(1)) if m else None

                latent_summaries: list[dict] = []

                for jsonl_path in sorted(latent_dir.glob("*.jsonl")):
                    if jsonl_path.stem.endswith(".bak"):
                        continue

                    records = [json.loads(l) for l in open(jsonl_path)]
                    filtered = [r for r in records
                                if _is_1152(r.get("id") or r.get("sample_id", ""))]

                    logger.info("Steering %s/%s: %d → %d",
                                latent_dir.name, jsonl_path.name,
                                len(records), len(filtered))

                    m_alpha = re.search(r"alpha_?(-?[\d.]+)", jsonl_path.stem)
                    alpha = float(m_alpha.group(1)) if m_alpha else 0.0

                    s = _steering_summary(filtered, alpha, latent_idx)
                    latent_summaries.append(s)
                    all_summaries.append(s)

                    if not dry_run:
                        bak = jsonl_path.with_suffix(".jsonl.bak")
                        if not bak.exists():
                            jsonl_path.rename(bak)
                        with open(jsonl_path, "w") as f:
                            for r in filtered:
                                f.write(json.dumps(r) + "\n")

                # Rewrite sweep_summary.json for this latent
                if latent_summaries and not dry_run:
                    sweep_path = latent_dir / "sweep_summary.json"
                    with open(sweep_path, "w") as f:
                        json.dump(latent_summaries, f, indent=2)
                    logger.info("Wrote %s", sweep_path)

            # Rewrite all_summaries.json for this layer
            if all_summaries and not dry_run:
                all_path = layer_dir / "all_summaries.json"
                with open(all_path, "w") as f:
                    json.dump(all_summaries, f, indent=2)
                logger.info("Wrote %s", all_path)


# ---------------------------------------------------------------------------
# EVA decoding helpers
# ---------------------------------------------------------------------------

def _eva_summary(records: list[dict], alpha: float, target_layer: int,
                 dataset: str) -> dict:
    n = len(records)
    if n == 0:
        return {}

    topic_stats: dict[str, dict] = defaultdict(lambda: {"vanilla_correct": 0,
                                                          "eva_correct": 0, "n": 0})
    bias_van = bias_eva = 0
    for r in records:
        topic_stats[r.get("topic", "unknown")]["n"] += 1
        topic_stats[r.get("topic", "unknown")]["vanilla_correct"] += int(
            r.get("is_correct_vanilla", False))
        topic_stats[r.get("topic", "unknown")]["eva_correct"] += int(
            r.get("is_correct_eva", False))
        bias_van += int(r.get("vanilla_matches_bias", False))
        bias_eva += int(r.get("eva_matches_bias", False))

    summary = {
        "alpha": alpha,
        "target_layer": target_layer,
        "dataset": dataset,
        "n": n,
        "vanilla_accuracy": sum(r.get("is_correct_vanilla", False) for r in records) / n,
        "eva_accuracy": sum(r.get("is_correct_eva", False) for r in records) / n,
        "accuracy_delta": (
            sum(r.get("is_correct_eva", False) for r in records)
            - sum(r.get("is_correct_vanilla", False) for r in records)
        ) / n,
        "vanilla_bias_ratio": bias_van / n,
        "eva_bias_ratio": bias_eva / n,
        "bias_ratio_delta": (bias_eva - bias_van) / n,
        "accuracy_by_topic": {
            topic: {
                "vanilla": d["vanilla_correct"] / d["n"],
                "eva": d["eva_correct"] / d["n"],
                "n": d["n"],
            }
            for topic, d in topic_stats.items()
        },
    }
    return summary


def patch_eva_vab(eva_root: Path, dry_run: bool) -> None:
    for model_dir in eva_root.iterdir():
        if not model_dir.is_dir():
            continue
        vab_dir = model_dir / "vlms_are_biased"
        if not vab_dir.exists():
            continue

        for layer_dir in vab_dir.iterdir():
            if not layer_dir.is_dir():
                continue
            m_layer = re.match(r"layer_(\d+)", layer_dir.name)
            target_layer = int(m_layer.group(1)) if m_layer else 0

            for alpha_dir in sorted(layer_dir.iterdir()):
                if not alpha_dir.is_dir():
                    continue
                results_path = alpha_dir / "results.jsonl"
                if not results_path.exists():
                    continue

                records = [json.loads(l) for l in open(results_path)]
                filtered = [r for r in records
                            if _is_1152(r.get("id") or r.get("sample_id", ""))]

                logger.info("EVA %s/%s: %d → %d",
                            layer_dir.name, alpha_dir.name,
                            len(records), len(filtered))

                m_alpha = re.search(r"alpha_(-?[\d.]+)", alpha_dir.name)
                alpha = float(m_alpha.group(1)) if m_alpha else 0.0

                if not dry_run:
                    bak = results_path.with_suffix(".jsonl.bak")
                    if not bak.exists():
                        results_path.rename(bak)
                    with open(results_path, "w") as f:
                        for r in filtered:
                            f.write(json.dumps(r) + "\n")

                    summary = _eva_summary(filtered, alpha, target_layer, "vlms_are_biased")
                    with open(alpha_dir / "summary.json", "w") as f:
                        json.dump(summary, f, indent=2)
                    logger.info("Wrote summary for %s/%s", layer_dir.name, alpha_dir.name)

            # Rewrite sweep_summary.json from individual summaries
            if not dry_run:
                sweep = []
                for alpha_dir in sorted(layer_dir.iterdir()):
                    s_path = alpha_dir / "summary.json"
                    if s_path.exists():
                        with open(s_path) as f:
                            sweep.append(json.load(f))
                if sweep:
                    with open(layer_dir / "sweep_summary.json", "w") as f:
                        json.dump(sweep, f, indent=2)
                    logger.info("Wrote sweep_summary for %s", layer_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Filter VAB results to 1152px only")
    parser.add_argument("--results_dir", default=None,
                        help="Path to results/ directory (default: repo_root/results/)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print what would be done without writing anything")
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    results_dir = Path(args.results_dir) if args.results_dir else repo_root / "results"

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    if args.dry_run:
        logger.info("DRY RUN — no files will be written")

    patch_revis_vab(results_dir / "revis", dry_run=args.dry_run)
    patch_steering_vab(results_dir / "steering", dry_run=args.dry_run)
    patch_eva_vab(results_dir / "eva_decoding", dry_run=args.dry_run)

    if args.dry_run:
        print("\nDry run complete — rerun without --dry_run to apply changes.")
    else:
        print("\nDone. Original files backed up as *.bak.")


if __name__ == "__main__":
    main()
