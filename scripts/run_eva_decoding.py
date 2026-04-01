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
from eva.eva_decoding import eva_decode_dataset

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


def _normalize_answer(s: str) -> str:
    """Strip curly-bracket formatting and lowercase.

    VAB prompts ask for answers in {Yes}/{No} format, but ground truth is
    stored as plain 'Yes'/'No'. Models may or may not include the brackets.
    """
    return s.strip().strip("{}").strip().lower()


def _vab_is_match(pred: str, target: str) -> bool:
    """Match prediction against target following lmms-eval VAB logic.

    1. Exact match after normalization.
    2. Fallback: extract digits and compare numerically.
    Mirrors vlms_are_biased_process_results in lmms-eval/tasks/vlms_are_biased/utils.py.
    """
    pred_n = _normalize_answer(pred)
    tgt_n = _normalize_answer(target)
    if pred_n == tgt_n:
        return True
    pred_digits = "".join(c for c in pred_n if c.isdigit())
    tgt_digits = "".join(c for c in tgt_n if c.isdigit())
    return bool(pred_digits) and bool(tgt_digits) and pred_digits == tgt_digits


def load_vlms_are_biased(dataset_id: str, split: str, n_samples: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset(dataset_id, split=split)
    samples = []
    for item in ds.select(range(min(n_samples, len(ds)))):
        samples.append({
            "id": item.get("ID") or len(samples),
            "image": item.get("image"),
            "messages": [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": item.get("prompt") or ""},
            ]}],
            "answer": _normalize_answer(item.get("ground_truth") or ""),
            "expected_bias": _normalize_answer(item.get("expected_bias") or ""),
            "topic": item.get("topic", ""),
            "sub_topic": item.get("sub_topic", ""),
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

    # Re-evaluate using lmms-eval-compatible metrics and attach expected_bias.
    # NOTE: main split contains each image at 3 resolutions — consistent with
    # how lmms-eval ran WS1, so we keep all triplicates rather than deduplicating.
    sample_by_id = {s["id"]: s for s in samples}
    topic_stats: dict[str, dict[str, int]] = {}

    for r in results:
        s = sample_by_id.get(r.get("id"), {})
        gt = s.get("answer", "")
        bias = s.get("expected_bias", "")
        r["expected_bias"] = bias
        r["topic"] = s.get("topic", "")

        # Recompute correctness with numeric fallback (matches lmms-eval logic)
        if gt:
            r["is_correct_vanilla"] = _vab_is_match(r["vanilla_answer"], gt)
            r["is_correct_eva"] = _vab_is_match(r["eva_answer"], gt)
        # bias_ratio: unconditional — does the model say the biased answer?
        # Lower is better (lmms-eval: higher_is_better: false)
        r["vanilla_matches_bias"] = _vab_is_match(r["vanilla_answer"], bias) if bias else False
        r["eva_matches_bias"] = _vab_is_match(r["eva_answer"], bias) if bias else False

        # Per-topic tracking
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
        "n": n,
        # Accuracy (lmms-eval compatible)
        "vanilla_accuracy": sum(r.get("is_correct_vanilla", False) for r in results) / n,
        "eva_accuracy": sum(r.get("is_correct_eva", False) for r in results) / n,
        "accuracy_delta": (
            sum(r.get("is_correct_eva", False) for r in results)
            - sum(r.get("is_correct_vanilla", False) for r in results)
        ) / n,
        # Bias ratio — unconditional, lower is better (mirrors lmms-eval)
        "vanilla_bias_ratio": sum(r.get("vanilla_matches_bias", False) for r in results) / n,
        "eva_bias_ratio": sum(r.get("eva_matches_bias", False) for r in results) / n,
        "bias_ratio_delta": (
            sum(r.get("eva_matches_bias", False) for r in results)
            - sum(r.get("vanilla_matches_bias", False) for r in results)
        ) / n,
        # Per-topic accuracy
        "accuracy_by_topic": {
            topic: {
                "vanilla": d["vanilla_correct"] / d["n"],
                "eva": d["eva_correct"] / d["n"],
                "n": d["n"],
            }
            for topic, d in topic_stats.items()
        },
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Exp 2.2: EVA Corrected Decoding")
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--dataset", default="vlms_are_biased",
                        choices=["vqav2", "vlms_are_biased"])
    parser.add_argument("--vab_dataset_id", default="anvo25/vlms-are-biased")
    parser.add_argument("--vab_split", default="main")
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
        print(
            f"α={alpha:.2f}  "
            f"acc: vanilla={summary.get('vanilla_accuracy', 0):.3f}  "
            f"eva={summary.get('eva_accuracy', 0):.3f}  "
            f"Δ={summary.get('accuracy_delta', 0):+.3f}  |  "
            f"bias_ratio: vanilla={summary.get('vanilla_bias_ratio', 0):.3f}  "
            f"eva={summary.get('eva_bias_ratio', 0):.3f}  "
            f"Δ={summary.get('bias_ratio_delta', 0):+.3f}"
        )

    # Save combined sweep summary
    with open(os.path.join(output_dir, "sweep_summary.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)


if __name__ == "__main__":
    main()
