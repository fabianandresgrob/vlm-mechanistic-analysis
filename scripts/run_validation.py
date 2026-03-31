"""
Script to run Exp 3.2: Statistical Validation on VLMs Are Biased.

Requires:
  - Exp 3.1 scores (results/feature_search/.../scores.npz, top_features.json)
  - Target layer

Usage:
    python scripts/run_validation.py \\
        --model google/gemma-3-4b-it \\
        --target_layer 20 \\
        --feature_search_dir results/feature_search/google_gemma-3-4b-it/layer_20 \\
        --output_dir results/validation/ \\
        --device cuda
"""
from __future__ import annotations
import argparse, json, logging, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chain_of_embedding.models.gemma3 import load_gemma3
from feature_search.sae_utils import load_sae, extract_answer_token_acts
from feature_search.validation import feature_activation_test

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_vab(dataset_id: str, split: str, n_samples: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset(dataset_id, split=split)
    samples = []
    for item in ds.select(range(min(n_samples, len(ds)))):
        cf_image = item.get("cf_image") or item.get("counterfactual_image")
        samples.append({
            "id": item.get("id") or len(samples),
            "image": item.get("image"),
            "cf_image": cf_image,
            "messages": [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": item.get("question") or item.get("query") or ""},
            ]}],
            "answer": item.get("answer") or "",
            "is_correct": float(item.get("correct", float("nan"))),
            "category": item.get("category", ""),
        })
    return samples


def main():
    parser = argparse.ArgumentParser(description="Exp 3.2: Statistical Validation")
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--model_size", default="4b")
    parser.add_argument("--target_layer", type=int, required=True)
    parser.add_argument("--width", default="16k")
    parser.add_argument("--l0_level", default="medium")
    parser.add_argument("--feature_search_dir", required=True,
                        help="Output dir from Exp 3.1 (contains top_features.json)")
    parser.add_argument("--vab_dataset_id", default="anvo25/vlms-are-biased")
    parser.add_argument("--vab_split", default="test")
    parser.add_argument("--n_samples", type=int, default=240)
    parser.add_argument("--n_top_features", type=int, default=20,
                        help="Number of top features from 3.1 to validate")
    parser.add_argument("--output_dir", default="results/validation/")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_slug = args.model.replace("/", "_")
    output_dir = os.path.join(repo_root, args.output_dir, model_slug,
                              f"layer_{args.target_layer}")
    os.makedirs(output_dir, exist_ok=True)

    # Load top features from Exp 3.1
    with open(os.path.join(args.feature_search_dir, "top_features.json")) as f:
        top_features = json.load(f)
    top_visual_indices = [f["latent_idx"] for f in top_features["top_visual"][:args.n_top_features]]

    # Load model + SAE
    model, processor = load_gemma3(args.model, device=args.device)
    sae = load_sae(args.target_layer, args.model_size, args.width, args.l0_level, args.device)

    # Load VAB — filter out samples with unknown correctness (nan)
    samples = load_vab(args.vab_dataset_id, args.vab_split, args.n_samples)
    samples = [s for s in samples if not np.isnan(s["is_correct"])]
    if not samples:
        logger.error("No samples with valid 'is_correct' labels after filtering. Check dataset.")
        sys.exit(1)
    logger.info("%d samples with valid correctness labels.", len(samples))
    is_correct = np.array([s["is_correct"] for s in samples])

    # Extract activations for condition B (original image) at target layer
    acts_vis, _ = extract_answer_token_acts(
        model, processor, samples, args.target_layer, args.device
    )

    # Encode through SAE
    x = torch.tensor(acts_vis, dtype=torch.float32, device=sae.device).unsqueeze(1)
    feat_acts_b = sae.encode(x)[:, 0, :].cpu().numpy()   # (n_samples, d_sae)

    # Statistical tests
    results = feature_activation_test(
        feat_acts_b, None, top_visual_indices, is_correct
    )

    with open(os.path.join(output_dir, "feature_tests.json"), "w") as f:
        json.dump(results, f, indent=2)

    summary = {
        "model_id": args.model, "target_layer": args.target_layer,
        "n_samples": int(len(samples)),
        "n_correct": int((is_correct == 1.0).sum()),
        "n_biased": int((is_correct == 0.0).sum()),
        "n_features_tested": len(results),
        "top_feature_by_effect": results[0] if results else None,
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTop features by effect size (correct vs biased on VAB):")
    for r in results[:10]:
        sig = "***" if r["pvalue"] < 0.001 else ("**" if r["pvalue"] < 0.01 else
              ("*" if r["pvalue"] < 0.05 else "ns"))
        print(f"  latent {r['latent_idx']:6d}  r={r['effect_size_r']:+.3f}  "
              f"p={r['pvalue']:.3e} {sig}")


if __name__ == "__main__":
    main()
