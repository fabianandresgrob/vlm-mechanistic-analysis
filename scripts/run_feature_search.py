"""
Script to run Exp 3.1: Contrastive SAE Feature Search.

Requires target_layer from Exp 2.4 + 2.5 intersection.

Usage:
    python scripts/run_feature_search.py \\
        --model google/gemma-3-4b-it \\
        --target_layer 20 \\
        --width 16k \\
        --n_samples 5000 \\
        --output_dir results/feature_search/ \\
        --device cuda

Outputs:
    results/feature_search/{model_slug}/layer_{N}/
        scores.npz      — f_vis, f_blind, s_visual, s_prior, noise_mask
        top_features.json — top-50 visual reliance + prior latents with scores
        summary.json    — metadata + top feature indices
"""
from __future__ import annotations
import argparse, json, logging, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chain_of_embedding.models.gemma3 import load_gemma3
from data_loaders import load_vqav2
from feature_search.sae_utils import load_sae, extract_answer_token_acts
from feature_search.contrastive_search import separation_scores

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Exp 3.1: Contrastive SAE Feature Search")
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--model_size", default="4b",
                        choices=["270m", "1b", "4b", "12b", "27b"])
    parser.add_argument("--target_layer", type=int, required=True,
                        help="Target LLM layer (from Exp 2.4 + 2.5 intersection)")
    parser.add_argument("--width", default="16k",
                        choices=["16k", "262k"])
    parser.add_argument("--l0_level", default="big",
                        choices=["small", "big"])
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--noise_threshold", type=float, default=0.02)
    parser.add_argument("--output_dir", default="results/feature_search/")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_slug = args.model.replace("/", "_")
    output_dir = os.path.join(
        repo_root, args.output_dir, model_slug, f"layer_{args.target_layer}"
    )
    os.makedirs(output_dir, exist_ok=True)

    acts_path = os.path.join(output_dir, "acts.npz")
    if os.path.exists(acts_path):
        logger.info("Loading cached activations from %s", acts_path)
        cached = np.load(acts_path)
        acts_vis, acts_blind = cached["acts_vis"], cached["acts_blind"]
    else:
        logger.info("Loading model %s…", args.model)
        model, processor = load_gemma3(args.model, device=args.device)

        logger.info("Loading %d VQAv2 samples…", args.n_samples)
        samples = load_vqav2(n_samples=args.n_samples)

        logger.info("Extracting activations at layer %d…", args.target_layer)
        acts_vis, acts_blind = extract_answer_token_acts(
            model, processor, samples, args.target_layer, args.device
        )
        np.savez(acts_path, acts_vis=acts_vis, acts_blind=acts_blind)

    logger.info("Loading SAE…")
    sae = load_sae(args.target_layer, args.model_size, args.width, args.l0_level, args.device)
    if sae is None:
        raise RuntimeError(f"SAE not available for layer {args.target_layer}")

    logger.info("Computing separation scores…")
    scores = separation_scores(
        acts_vis, acts_blind, sae,
        noise_baseline_acts=acts_blind,
        noise_threshold=args.noise_threshold,
    )

    np.savez(os.path.join(output_dir, "scores.npz"), **{
        k: v for k, v in scores.items() if isinstance(v, np.ndarray)
    })

    top_features = {
        "top_visual": [
            {"latent_idx": int(i), "s_visual": float(scores["s_visual"][i]),
             "f_vis": float(scores["f_vis"][i]), "f_blind": float(scores["f_blind"][i])}
            for i in scores["top_visual"][:50]
        ],
        "top_prior": [
            {"latent_idx": int(i), "s_prior": float(scores["s_prior"][i]),
             "f_vis": float(scores["f_vis"][i]), "f_blind": float(scores["f_blind"][i])}
            for i in scores["top_prior"][:50]
        ],
    }
    with open(os.path.join(output_dir, "top_features.json"), "w") as f:
        json.dump(top_features, f, indent=2)

    summary = {
        "model_id": args.model, "target_layer": args.target_layer,
        "width": args.width, "l0_level": args.l0_level,
        "n_samples": int(acts_vis.shape[0]),
        "n_sae_latents": int(sae.cfg.d_sae),
        "n_latents_after_noise_filter": int(scores["noise_mask"].sum()),
        "top_visual_latent": int(scores["top_visual"][0]),
        "top_visual_score": float(scores["s_visual"][scores["top_visual"][0]]),
        "top_prior_latent": int(scores["top_prior"][0]),
        "top_prior_score": float(scores["s_prior"][scores["top_prior"][0]]),
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTop visual reliance latents (layer {args.target_layer}):")
    for feat in top_features["top_visual"][:10]:
        print(f"  latent {feat['latent_idx']:6d}  s={feat['s_visual']:+.4f}  "
              f"f_vis={feat['f_vis']:.3f}  f_blind={feat['f_blind']:.3f}")


if __name__ == "__main__":
    main()
