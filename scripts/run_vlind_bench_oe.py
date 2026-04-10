"""Evaluate a vision-language model on vlind-bench-oe.

Loads the dataset from HuggingFace (fabiangrob/vlind-bench-oe), runs the
model on each sample, scores responses as 'correct', 'biased', or 'other',
and writes per-sample results to a JSONL file.

Scoring (from vlind_bench_oe_utils):
  correct — response matches an expected_answer (what the CF image shows)
  biased  — response matches a biased_answer (real-world language prior)
  other   — neither matched

Usage:
    python scripts/run_vlind_bench_oe.py \\
        --model google/gemma-3-4b-it \\
        --output results/vlind_bench_oe_gemma3_4b.jsonl

    # dry run (first 20 samples, no output file)
    python scripts/run_vlind_bench_oe.py \\
        --model google/gemma-3-4b-it \\
        --n_samples 20 --dry_run

    # different instruction / dataset split
    python scripts/run_vlind_bench_oe.py \\
        --model google/gemma-3-27b-it \\
        --instructions "\\nAnswer the question using a single word or phrase." \\
        --output results/vlind_bench_oe_gemma3_27b.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_creation.vlind_bench_oe_utils import score_response, format_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_INSTRUCTIONS = "\nAnswer the question using a single word or phrase."
DEFAULT_DATASET = "fabiangrob/vlind-bench-oe"


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

def _load_model(model_id: str, dtype: str = "bfloat16"):
    """Load a HuggingFace model + processor for VLM inference."""
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    logger.info("Loading model %s …", model_id)
    torch_dtype = getattr(torch, dtype)
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    logger.info("Model loaded on %s.", device)
    return model, processor


def _run_model(model, processor, image, prompt: str, max_new_tokens: int = 32) -> str:
    """Run one inference step, return the decoded model response."""
    import torch

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Decode only newly generated tokens
    input_len = inputs["input_ids"].shape[1]
    generated = output_ids[0, input_len:]
    return processor.decode(generated, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------

def _compute_metrics(results: list[dict]) -> dict:
    """Aggregate per-sample results into benchmark metrics."""
    total = len(results)
    if total == 0:
        return {}

    n_correct = sum(r["score"] == "correct" for r in results)
    n_biased = sum(r["score"] == "biased" for r in results)
    n_other = sum(r["score"] == "other" for r in results)

    metrics = {
        "n_total": total,
        "accuracy":    n_correct / total,
        "bias_rate":   n_biased / total,
        "other_rate":  n_other / total,
    }

    # Per-concept breakdown
    by_concept: dict[str, list[str]] = {}
    for r in results:
        c = r.get("concept", "unknown")
        by_concept.setdefault(c, []).append(r["score"])

    concept_metrics = {}
    for c, scores in sorted(by_concept.items()):
        n = len(scores)
        concept_metrics[c] = {
            "n": n,
            "accuracy":  sum(s == "correct" for s in scores) / n,
            "bias_rate": sum(s == "biased" for s in scores) / n,
        }
    metrics["per_concept"] = concept_metrics

    return metrics


def _print_metrics(metrics: dict) -> None:
    print("\n=== vlind-bench-oe results ===")
    print(f"  Total samples : {metrics['n_total']}")
    print(f"  Accuracy      : {metrics['accuracy']:.1%}  (correct)")
    print(f"  Bias rate     : {metrics['bias_rate']:.1%}  (biased)")
    print(f"  Other         : {metrics['other_rate']:.1%}")
    print("\nPer concept:")
    for concept, cm in metrics.get("per_concept", {}).items():
        print(f"  {concept:20s}  n={cm['n']:4d}  acc={cm['accuracy']:.1%}  bias={cm['bias_rate']:.1%}")
    print()


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    model_id: str,
    output_path: Path | None,
    dataset_name: str = DEFAULT_DATASET,
    instructions: str = DEFAULT_INSTRUCTIONS,
    n_samples: int | None = None,
    dtype: str = "bfloat16",
    dry_run: bool = False,
) -> list[dict]:
    from datasets import load_dataset

    logger.info("Loading dataset %s …", dataset_name)
    ds = load_dataset(dataset_name, split="train")
    if n_samples is not None:
        ds = ds.select(range(min(n_samples, len(ds))))
    logger.info("Evaluating %d samples.", len(ds))

    if not dry_run:
        model, processor = _load_model(model_id, dtype=dtype)
    else:
        model = processor = None
        logger.info("Dry run — model not loaded.")

    # Resumable: skip already-processed instance_id+cf_img_idx pairs
    done_keys: set[tuple[int, int]] = set()
    results: list[dict] = []
    if output_path and output_path.exists():
        with open(output_path) as f:
            for line in f:
                r = json.loads(line)
                results.append(r)
                done_keys.add((r["instance_id"], r["cf_img_idx"]))
        logger.info("Resuming from %d existing results.", len(results))

    out_file = None if (dry_run or output_path is None) else open(output_path, "a")

    try:
        for sample in ds:
            key = (sample["instance_id"], sample["cf_img_idx"])
            if key in done_keys:
                continue

            prompt = format_prompt(sample["question"], instructions)
            image = sample["cf_image"]  # PIL image

            if dry_run:
                response = "<dry-run>"
                score = "other"
            else:
                response = _run_model(model, processor, image, prompt)
                score = score_response(
                    response,
                    sample["expected_answers"],
                    sample["biased_answers"],
                )

            result = {
                "instance_id":      sample["instance_id"],
                "cf_img_idx":       sample["cf_img_idx"],
                "concept":          sample["concept"],
                "question":         sample["question"],
                "instructions":     instructions,
                "expected_answers": sample["expected_answers"],
                "biased_answers":   sample["biased_answers"],
                "response":         response,
                "score":            score,
            }
            results.append(result)

            if out_file:
                out_file.write(json.dumps(result) + "\n")
                out_file.flush()

    finally:
        if out_file:
            out_file.close()

    metrics = _compute_metrics(results)
    _print_metrics(metrics)

    if output_path:
        metrics_path = output_path.with_suffix(".metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Metrics saved to %s", metrics_path)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate a VLM on vlind-bench-oe.")
    parser.add_argument("--model", required=True, help="HuggingFace model ID.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSONL path. Omit for stdout-only.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help=f"HF dataset name (default: {DEFAULT_DATASET}).")
    parser.add_argument("--instructions", default=DEFAULT_INSTRUCTIONS,
                        help="Instruction suffix appended to each question.")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Limit evaluation to first N samples.")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="Model dtype (use float16 for V100).")
    parser.add_argument("--dry_run", action="store_true",
                        help="Skip model inference; check dataset loading only.")
    args = parser.parse_args()

    run_evaluation(
        model_id=args.model,
        output_path=args.output,
        dataset_name=args.dataset,
        instructions=args.instructions,
        n_samples=args.n_samples,
        dtype=args.dtype,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
