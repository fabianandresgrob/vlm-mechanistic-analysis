"""
Scan results/ and print a summary table of all completed experiments.

Usage:
    python scripts/results_table.py
    python scripts/results_table.py --results_dir /path/to/results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def find_summaries(results_dir: Path) -> list[tuple[str, dict]]:
    """Recursively find all summary.json files and return (rel_path, data) pairs."""
    summaries = []
    for path in sorted(results_dir.rglob("summary.json")):
        try:
            with open(path) as f:
                data = json.load(f)
            rel = str(path.relative_to(results_dir))
            summaries.append((rel, data))
        except Exception:
            pass
    return summaries


def format_table(summaries: list[tuple[str, dict]]) -> str:
    """Format summaries as a readable table."""
    if not summaries:
        return "(no results found)"

    lines = []

    # ---- EVA JS divergence results ----
    eva_rows = [(p, d) for p, d in summaries if "peak_layer" in d]
    if eva_rows:
        lines.append("── Exp 2.1: EVA JS Divergence ─────────────────────────────────")
        lines.append(f"  {'path':<45} {'model':<25} {'dataset':<15} {'peak_layer':>10} {'n':>6}")
        lines.append("  " + "-" * 105)
        for path, d in eva_rows:
            lines.append(
                f"  {path:<45} {d.get('model_id',''):<25} {d.get('dataset',''):<15} "
                f"{d.get('peak_layer','?'):>10} {d.get('n_samples_processed','?'):>6}"
            )
        lines.append("")

    # ---- SAE convergence results ----
    sae_rows = [(p, d) for p, d in summaries if "convergence_layer" in d]
    if sae_rows:
        lines.append("── Exp 2.5: SAE Convergence ────────────────────────────────────")
        lines.append(f"  {'path':<45} {'model':<25} {'width':<6} {'conv_layer':>10} {'n':>6}")
        lines.append("  " + "-" * 97)
        for path, d in sae_rows:
            lines.append(
                f"  {path:<45} {d.get('model_id',''):<25} {d.get('sae_width',''):<6} "
                f"{d.get('convergence_layer','?'):>10} {d.get('n_samples','?'):>6}"
            )
        lines.append("")

    # ---- Chain-of-embedding results ----
    coe_rows = [(p, d) for p, d in summaries if "vip_median" in d]
    if coe_rows:
        lines.append("── Exp 2.4: Chain-of-Embedding ─────────────────────────────────")
        lines.append(f"  {'path':<45} {'model':<25} {'dataset':<15} {'vip':>5} {'mean_tvi':>10} {'n':>6}")
        lines.append("  " + "-" * 110)
        for path, d in coe_rows:
            lines.append(
                f"  {path:<45} {d.get('model_id',''):<25} {d.get('dataset',''):<15} "
                f"{d.get('vip_median','?'):>5} {d.get('mean', float('nan')):>10.4f} "
                f"{d.get('n_samples','?'):>6}"
            )
        lines.append("")

    # ---- EVA decoding results ----
    eva_dec_rows = [(p, d) for p, d in summaries if "eva_accuracy" in d]
    if eva_dec_rows:
        lines.append("── Exp 2.2: EVA Decoding ───────────────────────────────────────")
        lines.append(f"  {'path':<45} {'alpha':>6} {'layer':>6} {'vanilla':>9} {'eva':>9} {'delta':>7} {'n':>6}")
        lines.append("  " + "-" * 95)
        for path, d in eva_dec_rows:
            lines.append(
                f"  {path:<45} {d.get('alpha', '?'):>6} {d.get('target_layer','?'):>6} "
                f"{d.get('vanilla_accuracy', float('nan')):>9.3f} "
                f"{d.get('eva_accuracy', float('nan')):>9.3f} "
                f"{d.get('delta', float('nan')):>+7.3f} "
                f"{d.get('n','?'):>6}"
            )
        lines.append("")

    # ---- Anything else ----
    known_keys = {"peak_layer", "convergence_layer", "vip_median", "eva_accuracy"}
    other_rows = [(p, d) for p, d in summaries
                  if not any(k in d for k in known_keys)]
    if other_rows:
        lines.append("── Other results ───────────────────────────────────────────────")
        for path, d in other_rows:
            lines.append(f"  {path}")
            for k, v in list(d.items())[:5]:
                lines.append(f"      {k}: {v}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Print a summary of all experiment results")
    parser.add_argument("--results_dir", default=None,
                        help="Path to results directory (default: repo_root/results/)")
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    results_dir = Path(args.results_dir) if args.results_dir else repo_root / "results"

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        sys.exit(0)

    summaries = find_summaries(results_dir)
    print(f"\nResults in {results_dir}  ({len(summaries)} summary files found)\n")
    print(format_table(summaries))


if __name__ == "__main__":
    main()
