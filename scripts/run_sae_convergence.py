"""
Script to run Exp 2.5: GemmaScope SAE Convergence Mapping on Gemma 3.

Usage:
    python scripts/run_sae_convergence.py \\
        --model google/gemma-3-4b-it \\
        --model_size 4b \\
        --n_samples 500 \\
        --layers 0,4,8,12,16,18,20,22,24 \\
        --output_dir results/sae_convergence/ \\
        --device cuda \\
        --resume

Outputs to output_dir/:
    convergence_profile.npz  — raw per-layer metrics arrays
    convergence_profile.png  — 2-panel visualization
    summary.json             — convergence layer, model_id, metadata
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chain_of_embedding.models.gemma3 import load_gemma3, num_llm_layers
from data_loaders import load_vqav2
from sae_convergence.convergence import (
    compute_layer_convergence_profile,
    find_convergence_layer,
    plot_convergence_profile,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_layers(layers_str: str, n_total: int) -> list[int]:
    """Parse comma-separated layer indices, handling 'all' and defaults."""
    if not layers_str or layers_str.lower() == "all":
        return list(range(n_total))
    try:
        return sorted(set(int(x.strip()) for x in layers_str.split(",")))
    except ValueError:
        raise ValueError(f"Invalid --layers value: {layers_str!r}")


def main():
    parser = argparse.ArgumentParser(description="Exp 2.5: GemmaScope SAE Convergence Mapping")
    parser.add_argument("--model", default="google/gemma-3-4b-it", help="HuggingFace model ID")
    parser.add_argument(
        "--model_size",
        default="4b",
        choices=["2b", "4b", "9b", "27b"],
        help="Model size suffix for SAE release name",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=500,
        help="Number of VQAv2 samples to probe (500 is sufficient)",
    )
    parser.add_argument(
        "--layers",
        default=None,
        help="Comma-separated layer indices to probe. Default: every other layer. "
             "E.g. '0,4,8,12,16,18,20,22,24'",
    )
    parser.add_argument(
        "--width",
        default="16k",
        choices=["16k", "65k", "262k", "1m"],
        help="SAE feature dictionary width. All-layers SAEs: '16k' or '262k'. "
             "Selected-layer SAEs only: '65k' or '1m'.",
    )
    parser.add_argument(
        "--l0_level",
        default="small",
        choices=["small", "big", "medium", "large"],
        help="SAE sparsity level. All-layers: 'small' (~10-20) or 'big' (~60-120). "
             "Selected-layer SAEs also support 'medium' and 'large'.",
    )
    parser.add_argument("--output_dir", default="results/sae_convergence/", help="Output directory")
    parser.add_argument("--device", default="cuda", help="Device: 'cuda', 'cpu', or 'auto'")
    parser.add_argument("--resume", action="store_true", help="Load cached results if they exist")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(repo_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # --- Load model to determine layer count ---
    logger.info("Loading model %s on %s…", args.model, args.device)
    model, processor = load_gemma3(model_id=args.model, device=args.device)
    n_total_layers = num_llm_layers(model)
    logger.info("Model has %d LLM layers.", n_total_layers)

    layers = parse_layers(args.layers or "", n_total_layers)
    # Default: every other layer, always include layers near expected convergence (16-22)
    if not args.layers:
        layers = sorted(set(list(range(0, n_total_layers, 2)) + list(range(14, min(24, n_total_layers)))))
    logger.info("Probing layers: %s", layers)

    # --- Load samples ---
    samples = load_vqav2(n_samples=args.n_samples)
    logger.info("Loaded %d samples.", len(samples))

    # --- Compute convergence profile ---
    profile = compute_layer_convergence_profile(
        model=model,
        processor=processor,
        samples=samples,
        model_size=args.model_size,
        layers=layers,
        device=args.device if args.device != "auto" else "cuda",
        width=args.width,
        l0_level=args.l0_level,
        output_dir=output_dir,
        resume=args.resume,
    )

    # --- Analysis ---
    conv_layer = find_convergence_layer(profile)
    logger.info("Convergence layer: %d (of %d)", conv_layer, n_total_layers)

    # --- Plot ---
    fig_path = os.path.join(output_dir, "convergence_profile.png")
    plot_convergence_profile(profile, fig_path)

    # --- Save summary ---
    result_layers = profile["layers"].tolist() if hasattr(profile["layers"], "tolist") else list(profile["layers"])
    summary = {
        "model_id": args.model,
        "model_size": args.model_size,
        "sae_release_format": f"gemma-scope-2-{args.model_size}-it-resid_post_all",
        "sae_width": args.width,
        "sae_l0_level": args.l0_level,
        "n_samples": int(profile.get("n_samples", args.n_samples)),
        "n_total_layers": n_total_layers,
        "probed_layers": result_layers,
        "convergence_layer": conv_layer,
        "visual_normalized_mse": [
            float(v) for v in profile["visual_normalized_mse"]
        ],
        "text_normalized_mse": [
            float(v) for v in profile["text_normalized_mse"]
        ],
        "visual_sparsity": [float(v) for v in profile["visual_sparsity"]],
        "text_sparsity": [float(v) for v in profile["text_sparsity"]],
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved summary to %s", summary_path)

    print(f"\nConvergence layer: {conv_layer} (of {n_total_layers})")
    min_nmse_idx = np.nanargmin(profile["visual_normalized_mse"])
    print(
        f"Min visual normalized MSE: {profile['visual_normalized_mse'][min_nmse_idx]:.4f} "
        f"at layer {result_layers[min_nmse_idx]}"
    )
    print(f"Figure saved to: {fig_path}")


if __name__ == "__main__":
    main()
