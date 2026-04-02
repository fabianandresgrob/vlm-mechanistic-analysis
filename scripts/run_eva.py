"""
Script to run Exp 2.1: EVA JS Divergence Analysis on Gemma 3.

Usage:
    python scripts/run_eva.py \\
        --model google/gemma-3-4b-it \\
        --dataset vqav2 \\
        --n_samples 5000 \\
        --output_dir results/eva/ \\
        --device cuda \\
        --resume

Outputs to output_dir/:
    js_per_layer.npz  — raw per-sample JS divergences + sample metadata
    summary.json      — peak layer, mean JS curve, Spearman correlations
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np

# Allow running as a script from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chain_of_embedding.models.gemma3 import load_gemma3
from data_loaders import load_vab, load_vab_pairs, load_vilp, load_vlind_bench, load_vqav2
from eva.js_divergence import (
    compute_cf_js_divergence,
    compute_layer_js_divergence,
    correlate_with_correctness,
    find_peak_layer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def try_attach_correctness(samples: list[dict], lmms_results_path: str, model_id: str) -> list[dict]:
    """Attempt to fill in is_correct from lmms-eval per-sample results."""
    # Results files are expected as {lmms_results_path}/{model_name}/vqav2_val/*.jsonl
    model_name = model_id.replace("/", "_")
    result_glob = os.path.join(lmms_results_path, model_name, "**", "*.jsonl")
    import glob
    import jsonlines

    files = glob.glob(result_glob, recursive=True)
    if not files:
        logger.info("No lmms-eval results found at %s — is_correct will be NaN.", result_glob)
        return samples

    correct_by_id: dict = {}
    for fpath in files:
        try:
            with jsonlines.open(fpath) as reader:
                for row in reader:
                    qid = row.get("question_id") or row.get("id")
                    if qid is not None:
                        correct_by_id[qid] = float(row.get("exact_match", row.get("correct", float("nan"))))
        except Exception as e:
            logger.warning("Could not read %s: %s", fpath, e)

    attached = 0
    for s in samples:
        if s["id"] in correct_by_id:
            s["is_correct"] = correct_by_id[s["id"]]
            attached += 1
    logger.info("Attached correctness labels for %d / %d samples.", attached, len(samples))
    return samples


def main():
    parser = argparse.ArgumentParser(description="Exp 2.1: EVA JS Divergence on Gemma 3")
    parser.add_argument("--model", default="google/gemma-3-4b-it", help="HuggingFace model ID")
    parser.add_argument(
        "--dataset",
        default="vqav2",
        choices=["vqav2", "vlms_are_biased", "vab_pairs", "vilp", "vlind"],
        help="Dataset to run on",
    )
    parser.add_argument("--n_samples", type=int, default=5000, help="Number of samples to process")
    parser.add_argument("--output_dir", default="results/eva/", help="Output directory")
    parser.add_argument("--device", default="cuda", help="Device: 'cuda', 'cpu', or 'auto'")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (currently only 1 supported)")
    parser.add_argument("--resume", action="store_true", help="Skip already-processed samples")
    parser.add_argument(
        "--mode",
        default="standard",
        choices=["standard", "cf_disc"],
        help=(
            "standard: JS(p_vis, p_blind) — original EVA methodology. "
            "cf_disc: JS(p_vis, p_cf) — discriminability between original and CF image."
        ),
    )
    parser.add_argument(
        "--lmms_results",
        default=None,
        help="Path to lmms-eval results for attaching correctness labels (optional — "
             "Spearman correlation is skipped if not provided)",
    )
    parser.add_argument(
        "--vab_dataset_id",
        default="anvo25/vlms-are-biased",
        help="HuggingFace dataset ID for VLMs Are Biased (default: anvo25/vlms-are-biased).",
    )
    parser.add_argument(
        "--vab_split",
        default="main",
        help="Split to use for VLMs Are Biased (default: main)",
    )
    args = parser.parse_args()

    # Resolve output dir relative to repo root
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(repo_root, args.output_dir, args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    # --- Load samples ---
    if args.dataset == "vqav2":
        samples = load_vqav2(n_samples=args.n_samples)
    elif args.dataset == "vlms_are_biased":
        samples = load_vab(
            dataset_id=args.vab_dataset_id, split=args.vab_split, n_samples=args.n_samples
        )
    elif args.dataset == "vab_pairs":
        samples = load_vab_pairs(n_samples=args.n_samples)
    elif args.dataset == "vilp":
        samples = load_vilp(n_samples=args.n_samples)
    elif args.dataset == "vlind":
        samples = load_vlind_bench(n_samples=args.n_samples)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    logger.info("Loaded %d samples.", len(samples))

    # Optionally attach correctness from lmms-eval
    if args.lmms_results:
        samples = try_attach_correctness(samples, args.lmms_results, args.model)
    else:
        # Try default paths.yaml location
        try:
            import yaml
            cfg_path = os.path.join(repo_root, "configs", "paths.yaml")
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            lmms_path = os.path.join(repo_root, cfg.get("lmms_eval_results_path", ""))
            if lmms_path and os.path.exists(lmms_path):
                samples = try_attach_correctness(samples, lmms_path, args.model)
        except Exception:
            pass

    # --- Load model ---
    logger.info("Loading model %s on %s…", args.model, args.device)
    model, processor = load_gemma3(model_id=args.model, device=args.device)

    device_str = args.device if args.device != "auto" else "cuda"

    if args.mode == "cf_disc":
        # --- CF discriminability: JS(p_vis, p_cf) ---
        logger.info("Computing CF discriminability JS divergence (JS(p_vis, p_cf))…")
        result = compute_cf_js_divergence(
            model=model,
            processor=processor,
            samples=samples,
            target_token_position="last",
            device=device_str,
            output_dir=output_dir,
            resume=args.resume,
        )

        js_per_layer = result["js_per_layer"]
        mean_js = result["mean_js"]
        n_layers = mean_js.shape[0]
        peak_layer = find_peak_layer(mean_js)

        summary = {
            "model_id": args.model,
            "dataset": args.dataset,
            "mode": "cf_disc",
            "n_samples_processed": int(js_per_layer.shape[0]),
            "n_layers": n_layers,
            "peak_layer": peak_layer,
            "mean_js_curve": mean_js.tolist(),
            "std_js_curve": result["std_js"].tolist(),
        }
        summary_path = os.path.join(output_dir, "summary_cf_disc.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Saved CF disc summary to %s", summary_path)

        print(f"\n[CF discriminability] Peak JS(p_vis, p_cf) layer: {peak_layer} (of {n_layers})")
        print(f"Mean JS at peak layer: {mean_js[peak_layer]:.4f} bits")

    else:
        # --- Standard EVA: JS(p_vis, p_blind) ---
        logger.info("Computing layer-wise JS divergence (standard EVA)…")
        result = compute_layer_js_divergence(
            model=model,
            processor=processor,
            samples=samples,
            target_token_position="last",
            device=device_str,
            batch_size=args.batch_size,
            output_dir=output_dir,
            resume=args.resume,
        )

        js_per_layer = result["js_per_layer"]
        mean_js = result["mean_js"]
        is_correct = result["is_correct"]
        n_layers = mean_js.shape[0]
        peak_layer = find_peak_layer(mean_js)

        logger.info("Peak JS divergence layer: %d (of %d)", peak_layer, n_layers)

        correlations = {}
        for li in [peak_layer, peak_layer - 2, peak_layer + 2]:
            if 0 <= li < n_layers and not np.all(np.isnan(is_correct)):
                rho, pval = correlate_with_correctness(js_per_layer, is_correct, li)
                correlations[f"layer_{li}"] = {"rho": rho, "pvalue": pval}

        summary = {
            "model_id": args.model,
            "dataset": args.dataset,
            "mode": "standard",
            "n_samples_processed": int(js_per_layer.shape[0]),
            "n_layers": n_layers,
            "peak_layer": peak_layer,
            "mean_js_curve": mean_js.tolist(),
            "std_js_curve": result["std_js"].tolist(),
            "correctness_correlations": correlations,
        }
        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Saved summary to %s", summary_path)

        print(f"\nPeak JS divergence layer: {peak_layer} (of {n_layers})")
        print(f"Mean JS at peak layer: {mean_js[peak_layer]:.4f} bits")
        if correlations:
            for k, v in correlations.items():
                print(f"  Spearman rho at {k}: {v['rho']:.3f} (p={v['pvalue']:.3e})")


if __name__ == "__main__":
    main()
