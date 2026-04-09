"""
Script to run Exp 2.2: EVA-style corrected decoding.

Requires target_layer from Exp 2.1 (peak JS divergence layer).

Usage:
    python scripts/run_eva_decoding.py \\
        --model google/gemma-3-4b-it \\
        --dataset vlms_are_biased \\
        --target_layer 18 \\
        --alpha 1.0 \\
        --output_dir results/eva_decoding/ \\
        --device cuda

Sweeping alpha (for ablation):
    python scripts/run_eva_decoding.py \\
        --model google/gemma-3-4b-it \\
        --dataset vlms_are_biased \\
        --target_layer 18 \\
        --alpha_sweep 0.0,0.5,1.0,2.0,5.0 \\
        --output_dir results/eva_decoding/

Outputs:
    {output_dir}/{model_slug}/{dataset}/layer_{N}/alpha_{alpha}/
        results.jsonl   — per-sample vanilla vs EVA answers
        summary.json    — accuracy (+ bias_ratio for VAB)
"""

from __future__ import annotations

import argparse
import json
import jsonlines
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chain_of_embedding.models.gemma3 import load_gemma3
from data_loaders import (get_is_match, load_vab, load_vilp_expanded,
                          load_vlind_bench_lp, load_vqav2, compute_vlind_metrics)
from eva.eva_decoding import eva_decode_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_alpha(model, processor, samples, dataset, target_layer, alpha,
              output_dir, device, max_new_tokens):
    alpha_dir = os.path.join(output_dir, f"alpha_{alpha:.2f}")
    os.makedirs(alpha_dir, exist_ok=True)

    summary_path = os.path.join(alpha_dir, "summary.json")
    if os.path.exists(summary_path):
        logger.info("alpha=%.2f already done, skipping.", alpha)
        with open(summary_path) as f:
            return json.load(f)

    results = eva_decode_dataset(
        model, processor, samples, target_layer, alpha, device, max_new_tokens
    )

    is_match = get_is_match(dataset)
    sample_by_id = {s["id"]: s for s in samples}

    # VAB-specific per-topic and bias tracking
    topic_stats: dict[str, dict[str, int]] = {}

    for r in results:
        s = sample_by_id.get(r.get("id"), {})
        gt = s.get("answer", "")

        if gt:
            r["is_correct_vanilla"] = is_match(r["vanilla_answer"], gt)
            r["is_correct_eva"] = is_match(r["eva_answer"], gt)

        # Copy pipeline fields for VLind metric re-evaluation
        for field in ("instance_id", "stage", "qid", "cf_img_idx"):
            if field in s:
                r[field] = s[field]

        # VAB-only: bias ratio
        if dataset == "vlms_are_biased":
            bias = s.get("expected_bias", "")
            r["expected_bias"] = bias
            r["topic"] = s.get("topic", "")
            r["vanilla_matches_bias"] = is_match(r["vanilla_answer"], bias) if bias else False
            r["eva_matches_bias"] = is_match(r["eva_answer"], bias) if bias else False

            topic = r["topic"] or "unknown"
            if topic not in topic_stats:
                topic_stats[topic] = {"vanilla_correct": 0, "eva_correct": 0, "n": 0}
            topic_stats[topic]["n"] += 1
            topic_stats[topic]["vanilla_correct"] += int(r.get("is_correct_vanilla", False))
            topic_stats[topic]["eva_correct"] += int(r.get("is_correct_eva", False))

    with jsonlines.open(os.path.join(alpha_dir, "results.jsonl"), "w") as writer:
        writer.write_all(results)

    n = len(results)
    summary = {
        "alpha": alpha,
        "target_layer": target_layer,
        "dataset": dataset,
        "n": n,
        "vanilla_accuracy": sum(r.get("is_correct_vanilla", False) for r in results) / n,
        "eva_accuracy": sum(r.get("is_correct_eva", False) for r in results) / n,
        "accuracy_delta": (
            sum(r.get("is_correct_eva", False) for r in results)
            - sum(r.get("is_correct_vanilla", False) for r in results)
        ) / n,
    }

    if dataset == "vlind":
        # Compute accuracy_lp separately for vanilla and EVA conditions.
        # s_lp (conditional on CK+VP+CB) requires a full 4-stage eval run;
        # accuracy_lp is the raw LP-item accuracy and is computed here.
        van_results = [{"instance_id": r["instance_id"], "stage": "lp",
                        "qid": r["qid"], "cf_img_idx": r["cf_img_idx"],
                        "is_correct": r.get("is_correct_vanilla", False)}
                       for r in results if "instance_id" in r]
        eva_results = [{"instance_id": r["instance_id"], "stage": "lp",
                        "qid": r["qid"], "cf_img_idx": r["cf_img_idx"],
                        "is_correct": r.get("is_correct_eva", False)}
                       for r in results if "instance_id" in r]
        summary["vlind_metric"] = "accuracy_lp"
        summary["vanilla_accuracy_lp"] = compute_vlind_metrics(van_results)["accuracy_lp"]
        summary["eva_accuracy_lp"]     = compute_vlind_metrics(eva_results)["accuracy_lp"]
        summary["accuracy_lp_delta"]   = summary["eva_accuracy_lp"] - summary["vanilla_accuracy_lp"]

    if dataset == "vlms_are_biased":
        summary["vanilla_bias_ratio"] = sum(r.get("vanilla_matches_bias", False) for r in results) / n
        summary["eva_bias_ratio"] = sum(r.get("eva_matches_bias", False) for r in results) / n
        summary["bias_ratio_delta"] = (
            sum(r.get("eva_matches_bias", False) for r in results)
            - sum(r.get("vanilla_matches_bias", False) for r in results)
        ) / n
        summary["accuracy_by_topic"] = {
            topic: {
                "vanilla": d["vanilla_correct"] / d["n"],
                "eva": d["eva_correct"] / d["n"],
                "n": d["n"],
            }
            for topic, d in topic_stats.items()
        }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Exp 2.2: EVA Corrected Decoding")
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--dataset", default="vlms_are_biased",
                        choices=["vqav2", "vlms_are_biased", "vilp", "vlind"])
    parser.add_argument("--vab_dataset_id", default="anvo25/vlms-are-biased")
    parser.add_argument("--vab_split", default="main")
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--target_layer", type=int, required=True,
                        help="Peak JS divergence layer from Exp 2.1")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="EVA correction strength")
    parser.add_argument("--alpha_sweep", default=None,
                        help="Comma-separated alpha values, e.g. '0.0,0.5,1.0,2.0'")
    parser.add_argument("--max_new_tokens", type=int, default=3)
    parser.add_argument("--output_dir", default="results/eva_decoding/")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vilp_mode", default="without_fact",
                        choices=["without_fact", "with_fact"],
                        help="ViLP prompt mode (default: without_fact)")
    parser.add_argument("--vilp_images", default="cf_only",
                        choices=["all", "cf_only"],
                        help="ViLP image subset: cf_only=images 2+3 (default), all=all 3")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_slug = args.model.replace("/", "_")
    output_dir = os.path.join(
        repo_root, args.output_dir, model_slug, args.dataset,
        f"layer_{args.target_layer}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Load samples
    n = args.n_samples or None
    if args.dataset == "vqav2":
        samples = load_vqav2(n_samples=n)
    elif args.dataset == "vlms_are_biased":
        samples = load_vab(dataset_id=args.vab_dataset_id, split=args.vab_split, n_samples=n, resolution=1152)
    elif args.dataset == "vilp":
        samples = load_vilp_expanded(n_samples=n, mode=args.vilp_mode, images=args.vilp_images)
    elif args.dataset == "vlind":
        samples = load_vlind_bench_lp(n_samples=n)
    logger.info("Loaded %d samples.", len(samples))

    logger.info("Loading %s…", args.model)
    model, processor = load_gemma3(args.model, device=args.device)

    alphas = [float(a) for a in args.alpha_sweep.split(",")] if args.alpha_sweep else [args.alpha]

    all_summaries = []
    for alpha in alphas:
        logger.info("Running EVA decode with alpha=%.2f…", alpha)
        summary = run_alpha(
            model, processor, samples, args.dataset, args.target_layer,
            alpha, output_dir, args.device, args.max_new_tokens,
        )
        all_summaries.append(summary)
        line = (f"α={alpha:.2f}  vanilla={summary.get('vanilla_accuracy', 0):.3f}  "
                f"eva={summary.get('eva_accuracy', 0):.3f}  "
                f"Δ={summary.get('accuracy_delta', 0):+.3f}")
        if "vanilla_bias_ratio" in summary:
            line += (f"  |  bias: vanilla={summary['vanilla_bias_ratio']:.3f}  "
                     f"eva={summary['eva_bias_ratio']:.3f}  "
                     f"Δ={summary['bias_ratio_delta']:+.3f}")
        print(line)

    with open(os.path.join(output_dir, "sweep_summary.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)


if __name__ == "__main__":
    main()
