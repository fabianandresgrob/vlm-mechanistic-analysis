"""
REVIS: Sparse Latent Steering to Mitigate Visual Grounding Failures.

Adapted from arXiv:2602.11824 for Gemma 3 + VAB/ViLP benchmarks.

Two stages:

  Stage 1 — extract vector (--mode extract):
    Compute the REVIS steering vector for a given layer from calibration data
    and save it to output_dir/revis_vector_layer{N}.pt (+ metadata JSON).

  Stage 2 — evaluate steering (--mode steer):
    Load a saved vector and run steered generation on the benchmark,
    sweeping over alpha values. Outputs per-sample JSONL files.

Usage:
    # Extract vector at layer 22 from 100 VAB calibration samples
    python scripts/run_revis.py --mode extract \\
        --model google/gemma-3-4b-it --layer 22 \\
        --n_calib 100 --output_dir results/revis/ --device cuda

    # Steer on VAB with the extracted vector
    python scripts/run_revis.py --mode steer \\
        --model google/gemma-3-4b-it --layer 22 \\
        --vector results/revis/revis_vector_layer22.pt \\
        --dataset vab --n_samples 240 \\
        --alphas -200,-100,-50,0,50,100,200 \\
        --output_dir results/revis/ --device cuda

    # Run both stages at once
    python scripts/run_revis.py --mode both \\
        --model google/gemma-3-4b-it --layer 22 \\
        --output_dir results/revis/ --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chain_of_embedding.models.gemma3 import load_gemma3
from data_loaders import load_vab, load_vqav2
from feature_search.steering import steering_hook, steered_generate
from revis.vector_calculator import compute_revis_vector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_calibration_samples(dataset: str, n_samples: int) -> list[dict]:
    """Load calibration samples for vector extraction."""
    if dataset == "vab":
        return load_vab(n_samples=n_samples)
    elif dataset == "vqav2":
        return load_vqav2(n_samples=n_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}. Choose 'vab' or 'vqav2'.")


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_extract(args, model, processor) -> torch.Tensor:
    """Extract and save the REVIS steering vector."""
    samples = load_calibration_samples(args.dataset, args.n_calib)
    logger.info("Extracting REVIS vector at layer %d from %d samples…", args.layer, len(samples))

    v_pure, metadata = compute_revis_vector(
        model=model,
        processor=processor,
        samples=samples,
        layer_idx=args.layer,
        device=args.device,
        normalize=True,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    vec_path = os.path.join(args.output_dir, f"revis_vector_layer{args.layer}.pt")
    meta_path = os.path.join(args.output_dir, f"revis_vector_layer{args.layer}_meta.json")

    torch.save(v_pure.cpu(), vec_path)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved vector to %s", vec_path)
    logger.info(
        "Diagnostics: cos(visual, lang) before=%.4f → after=%.4f",
        metadata["cos_visual_lang_before"],
        metadata["cos_visual_lang_after"],
    )
    return v_pure


def stage_steer(args, model, processor, steering_vector: torch.Tensor):
    """Run steered generation sweep and write per-alpha JSONL files."""
    samples = load_calibration_samples(args.dataset, args.n_samples)
    alphas = [float(a) for a in args.alphas.split(",")]
    logger.info(
        "Steering on %d samples, layer %d, alphas=%s",
        len(samples), args.layer, alphas,
    )

    sv = steering_vector.to(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Pre-compute vanilla answers once (avoid N*K forward passes)
    logger.info("Pre-computing vanilla answers…")
    vanilla_answers: dict[int, str] = {}
    for sample in samples:
        sid = sample["id"]
        result = steered_generate(
            model, processor, sample,
            target_layer=args.layer,
            steering_vector=sv,
            alpha=0.0,
            device=args.device,
            max_new_tokens=20,
        )
        vanilla_answers[sid] = result["vanilla_answer"]

    # Sweep alphas
    for alpha in alphas:
        if alpha == 0.0:
            continue
        out_path = os.path.join(args.output_dir, f"revis_layer{args.layer}_alpha{alpha:g}.jsonl")
        if args.resume and os.path.exists(out_path):
            logger.info("Skipping alpha=%g (already exists).", alpha)
            continue

        records = []
        for sample in samples:
            sid = sample["id"]
            result = steered_generate(
                model, processor, sample,
                target_layer=args.layer,
                steering_vector=sv,
                alpha=alpha,
                device=args.device,
                max_new_tokens=20,
                precomputed_vanilla=vanilla_answers[sid],
            )
            result["sample_id"] = sid
            records.append(result)

        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        n_correct = sum(r.get("is_correct_steered", False) for r in records)
        n_vanilla = sum(r.get("is_correct_vanilla", False) for r in records)
        logger.info(
            "alpha=%6.1f | steered acc=%.3f  vanilla acc=%.3f",
            alpha, n_correct / len(records), n_vanilla / len(records),
        )

    # Write vanilla baseline
    vanilla_path = os.path.join(args.output_dir, f"revis_layer{args.layer}_alpha0.jsonl")
    with open(vanilla_path, "w") as f:
        for sample in samples:
            sid = sample["id"]
            r = {
                "sample_id": sid,
                "vanilla_answer": vanilla_answers[sid],
                "steered_answer": vanilla_answers[sid],
                "alpha": 0.0,
                "target_layer": args.layer,
            }
            if "answer" in sample and sample["answer"]:
                r["is_correct_vanilla"] = vanilla_answers[sid].lower() == sample["answer"].strip().lower()
                r["is_correct_steered"] = r["is_correct_vanilla"]
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="REVIS steering for Gemma 3")
    parser.add_argument("--mode", choices=["extract", "steer", "both"], default="both")
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--layer", type=int, required=True, help="Target LLM layer (0-based)")
    parser.add_argument("--dataset", default="vab", choices=["vab", "vqav2"])
    parser.add_argument("--n_calib", type=int, default=100, help="Calibration samples for vector extraction")
    parser.add_argument("--n_samples", type=int, default=240, help="Evaluation samples for steering")
    parser.add_argument("--alphas", default="-200,-100,-50,0,50,100,200",
                        help="Comma-separated alpha sweep values")
    parser.add_argument("--vector", default=None,
                        help="Path to pre-computed .pt vector (required for --mode steer)")
    parser.add_argument("--output_dir", default="results/revis/")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    args.output_dir = os.path.join(repo_root, args.output_dir)

    logger.info("Loading model %s on %s…", args.model, args.device)
    model, processor = load_gemma3(model_id=args.model, device=args.device)

    if args.mode in ("extract", "both"):
        v_pure = stage_extract(args, model, processor)
    else:
        if not args.vector:
            parser.error("--vector is required for --mode steer")
        v_pure = torch.load(args.vector, map_location=args.device)
        logger.info("Loaded vector from %s", args.vector)

    if args.mode in ("steer", "both"):
        stage_steer(args, model, processor, v_pure)


if __name__ == "__main__":
    main()
