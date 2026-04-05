"""
Script to run Exp 3.3: Causal Steering Intervention.

Requires:
  - Top visual reliance latents from Exp 3.1 (top_features.json)
  - Target layer

Usage:
    python scripts/run_steering.py \\
        --model google/gemma-3-4b-it \\
        --target_layer 20 \\
        --latent_idx 1234 \\
        --alpha_sweep "-500,-200,-100,0,100,200,500" \\
        --dataset vlms_are_biased \\
        --output_dir results/steering/ \\
        --device cuda

Or sweep top-N features automatically:
    python scripts/run_steering.py \\
        --model google/gemma-3-4b-it \\
        --target_layer 20 \\
        --feature_search_dir results/feature_search/google_gemma-3-4b-it/layer_20 \\
        --n_top_features 5 \\
        --alpha_sweep "0,100,200,500" \\
        --dataset vlms_are_biased \\
        --output_dir results/steering/
"""
from __future__ import annotations
import argparse, json, logging, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chain_of_embedding.models.gemma3 import load_gemma3
from data_loaders import load_vab, load_vilp, load_vlind_bench, get_is_match
from feature_search.sae_utils import load_sae
from feature_search.steering import get_steering_vector, steered_generate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_dataset(dataset: str, n_samples: int) -> list[dict]:
    if dataset == "vab":
        return load_vab(n_samples=n_samples)
    elif dataset == "vilp":
        return load_vilp(n_samples=n_samples)
    elif dataset == "vlind":
        return load_vlind_bench(n_samples=n_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")


def run_one_latent_alpha(model, processor, samples, target_layer,
                          steering_vector, alpha, output_dir, device,
                          vanilla_cache: dict | None = None,
                          is_match_fn=None):
    import jsonlines
    from tqdm import tqdm
    results = []
    for sample in tqdm(samples, desc=f"Steering α={alpha}"):
        try:
            r = steered_generate(model, processor, sample, target_layer,
                                  steering_vector, alpha, device,
                                  precomputed_vanilla=vanilla_cache.get(sample.get("id")) if vanilla_cache else None,
                                  is_match_fn=is_match_fn)
            r["id"] = sample.get("id")
            r["category"] = sample.get("category", "")
            results.append(r)
        except Exception as e:
            logger.warning("Sample %s failed: %s", sample.get("id"), e)

    n = len(results)
    vanilla_acc = sum(r.get("is_correct_vanilla", False) for r in results) / n if n > 0 else 0.0
    steered_acc = sum(r.get("is_correct_steered", False) for r in results) / n if n > 0 else 0.0

    with jsonlines.open(os.path.join(output_dir, f"alpha_{alpha:g}.jsonl"), "w") as w:
        w.write_all(results)

    return {"alpha": alpha, "vanilla_accuracy": vanilla_acc,
            "steered_accuracy": steered_acc,
            "delta": steered_acc - vanilla_acc, "n": n}


def main():
    parser = argparse.ArgumentParser(description="Exp 3.3: Causal Steering")
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--model_size", default="4b")
    parser.add_argument("--target_layer", type=int, required=True)
    parser.add_argument("--width", default="16k")
    parser.add_argument("--l0_level", default="big", choices=["small", "big"])
    parser.add_argument("--latent_idx", type=int, default=None,
                        help="Specific latent to steer. If None, use --feature_search_dir.")
    parser.add_argument("--feature_search_dir", default=None)
    parser.add_argument("--n_top_features", type=int, default=3)
    parser.add_argument("--alpha_sweep", default="0,100,200,500",
                        help="Comma-separated alpha values")
    parser.add_argument("--dataset", default="vab", choices=["vab", "vilp", "vlind"])
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Samples to evaluate (default: all available)")
    parser.add_argument("--output_dir", default="results/steering/")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_slug = args.model.replace("/", "_")
    base_output_dir = os.path.join(repo_root, args.output_dir, model_slug,
                                   args.dataset, f"layer_{args.target_layer}")
    os.makedirs(base_output_dir, exist_ok=True)

    alphas = [float(a) for a in args.alpha_sweep.split(",")]

    # Determine which latents to steer
    if args.latent_idx is not None:
        latent_indices = [args.latent_idx]
    elif args.feature_search_dir:
        with open(os.path.join(args.feature_search_dir, "top_features.json")) as f:
            top = json.load(f)
        if "top_visual" not in top:
            parser.error(f"top_features.json missing 'top_visual' key. Keys found: {list(top.keys())}")
        latent_indices = [f["latent_idx"] for f in top["top_visual"][:args.n_top_features]]
    else:
        parser.error("Either --latent_idx or --feature_search_dir is required.")

    model, processor = load_gemma3(args.model, device=args.device)
    sae = load_sae(args.target_layer, args.model_size, args.width, args.l0_level, args.device)
    samples = load_dataset(args.dataset, args.n_samples)
    is_match_fn = get_is_match(args.dataset)

    # Pre-compute vanilla answers once (shared across all latents and alphas)
    logger.info("Pre-computing vanilla answers for %d samples...", len(samples))
    import torch
    from tqdm import tqdm as _tqdm
    vanilla_cache: dict = {}
    _dummy_sv = get_steering_vector(sae, latent_indices[0])
    with torch.no_grad():
        for s in _tqdm(samples, desc="Vanilla"):
            try:
                r = steered_generate(model, processor, s, args.target_layer,
                                     _dummy_sv, alpha=0.0, device=args.device,
                                     is_match_fn=is_match_fn)
                vanilla_cache[s.get("id")] = r["vanilla_answer"]
            except Exception as e:
                logger.warning("Vanilla pass failed for sample %s: %s", s.get("id"), e)

    all_summaries = []
    for latent_idx in latent_indices:
        sv = get_steering_vector(sae, latent_idx)
        latent_dir = os.path.join(base_output_dir, f"latent_{latent_idx}")
        os.makedirs(latent_dir, exist_ok=True)

        latent_summaries = []
        for alpha in alphas:
            summary = run_one_latent_alpha(
                model, processor, samples, args.target_layer,
                sv, alpha, latent_dir, args.device,
                vanilla_cache=vanilla_cache,
                is_match_fn=is_match_fn,
            )
            summary["latent_idx"] = latent_idx
            latent_summaries.append(summary)
            print(f"  latent {latent_idx}  α={alpha:+.0f}  "
                  f"vanilla={summary['vanilla_accuracy']:.3f}  "
                  f"steered={summary['steered_accuracy']:.3f}  "
                  f"Δ={summary['delta']:+.3f}")

        with open(os.path.join(latent_dir, "sweep_summary.json"), "w") as f:
            json.dump(latent_summaries, f, indent=2)
        all_summaries.extend(latent_summaries)

    with open(os.path.join(base_output_dir, "all_summaries.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)


if __name__ == "__main__":
    main()
