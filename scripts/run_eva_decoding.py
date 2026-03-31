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
    {output_dir}/{model_slug}/{dataset}/alpha_{alpha}/
        results.jsonl   — per-sample vanilla vs EVA answers
        summary.json    — accuracy delta table
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
from eva.eva_decoding import accuracy_summary, eva_decode_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_vqav2(n_samples: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("lmms-lab/VQAv2", split="validation")
    samples = []
    for item in ds.select(range(min(n_samples, len(ds)))):
        samples.append({
            "id": item.get("question_id", len(samples)),
            "image": item["image"],
            "messages": [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": item["question"]},
            ]}],
            "answer": item.get("multiple_choice_answer", ""),
        })
    return samples


def load_vlms_are_biased(dataset_id: str, split: str, n_samples: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset(dataset_id, split=split)
    samples = []
    for item in ds.select(range(min(n_samples, len(ds)))):
        samples.append({
            "id": item.get("id") or len(samples),
            "image": item.get("image"),
            "messages": [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": item.get("question") or item.get("query") or ""},
            ]}],
            "answer": item.get("answer") or item.get("gt_answer") or "",
            "category": item.get("category", ""),
        })
    return samples


def run_alpha(model, processor, samples, target_layer, alpha, output_dir, device, max_new_tokens):
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

    with jsonlines.open(os.path.join(alpha_dir, "results.jsonl"), "w") as writer:
        writer.write_all(results)

    summary = {"alpha": alpha, "target_layer": target_layer, **accuracy_summary(results)}
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Exp 2.2: EVA Corrected Decoding")
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--dataset", default="vlms_are_biased",
                        choices=["vqav2", "vlms_are_biased"])
    parser.add_argument("--vab_dataset_id", default="anvo25/vlms-are-biased")
    parser.add_argument("--vab_split", default="test")
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--target_layer", type=int, required=True,
                        help="Peak JS divergence layer from Exp 2.1")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="EVA correction strength")
    parser.add_argument("--alpha_sweep", default=None,
                        help="Comma-separated alpha values for ablation, e.g. '0.0,0.5,1.0,2.0'")
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--output_dir", default="results/eva_decoding/")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_slug = args.model.replace("/", "_")
    output_dir = os.path.join(
        repo_root, args.output_dir, model_slug, args.dataset,
        f"layer_{args.target_layer}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Load samples
    n = args.n_samples or 999999
    if args.dataset == "vqav2":
        samples = load_vqav2(n)
    else:
        samples = load_vlms_are_biased(args.vab_dataset_id, args.vab_split, n)
    logger.info("Loaded %d samples.", len(samples))

    # Load model
    logger.info("Loading %s…", args.model)
    model, processor = load_gemma3(args.model, device=args.device)

    # Determine alpha values to run
    if args.alpha_sweep:
        alphas = [float(a) for a in args.alpha_sweep.split(",")]
    else:
        alphas = [args.alpha]

    all_summaries = []
    for alpha in alphas:
        logger.info("Running EVA decode with alpha=%.2f…", alpha)
        summary = run_alpha(
            model, processor, samples, args.target_layer, alpha,
            output_dir, args.device, args.max_new_tokens,
        )
        all_summaries.append(summary)
        print(f"α={alpha:.2f}  vanilla={summary.get('vanilla_accuracy', 'n/a'):.3f}  "
              f"eva={summary.get('eva_accuracy', 'n/a'):.3f}  "
              f"Δ={summary.get('delta', 'n/a'):+.3f}")

    # Save combined sweep summary
    with open(os.path.join(output_dir, "sweep_summary.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)


if __name__ == "__main__":
    main()
