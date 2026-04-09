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
from data_loaders import load_vab, load_vilp_expanded, load_vlind_bench_lp, get_is_match
from feature_search.sae_utils import load_sae
from feature_search.steering import get_steering_vector, get_combined_steering_vector, steered_generate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_dataset(dataset: str, n_samples: int,
                  vilp_mode: str = "without_fact",
                  vilp_images: str = "cf_only") -> list[dict]:
    if dataset == "vab":
        return load_vab(n_samples=n_samples)
    elif dataset == "vilp":
        return load_vilp_expanded(n_samples=n_samples, mode=vilp_mode, images=vilp_images)
    elif dataset == "vlind":
        return load_vlind_bench_lp(n_samples=n_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")


def run_one_latent_alpha(model, processor, samples, target_layer,
                          steering_vector, alpha, output_dir, device,
                          vanilla_cache: dict | None = None,
                          is_match_fn=None, skip_existing: bool = False):
    import jsonlines
    from tqdm import tqdm

    out_path = os.path.join(output_dir, f"alpha_{alpha:g}.jsonl")
    if skip_existing and os.path.exists(out_path):
        records = [json.loads(l) for l in open(out_path)]
        n = len(records)
        van = sum(r.get("is_correct_vanilla", False) for r in records) / n if n > 0 else 0.0
        ste = sum(r.get("is_correct_steered", False) for r in records) / n if n > 0 else 0.0
        logger.info("Skipping existing %s (vanilla=%.3f steered=%.3f)", out_path, van, ste)
        return {"alpha": alpha, "vanilla_accuracy": van, "steered_accuracy": ste,
                "delta": ste - van, "n": n}

    results = []
    for sample in tqdm(samples, desc=f"Steering α={alpha}"):
        try:
            r = steered_generate(model, processor, sample, target_layer,
                                  steering_vector, alpha, device,
                                  precomputed_vanilla=vanilla_cache.get(sample.get("id")) if vanilla_cache else None,
                                  is_match_fn=is_match_fn)
            r["id"] = sample.get("id")
            r["category"] = sample.get("category", "")
            for field in ("instance_id", "stage", "qid", "cf_img_idx"):
                if field in sample:
                    r[field] = sample[field]
            results.append(r)
        except Exception as e:
            logger.warning("Sample %s failed: %s", sample.get("id"), e)

    n = len(results)
    vanilla_acc = sum(r.get("is_correct_vanilla", False) for r in results) / n if n > 0 else 0.0
    steered_acc = sum(r.get("is_correct_steered", False) for r in results) / n if n > 0 else 0.0

    with jsonlines.open(out_path, "w") as w:
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
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip alpha/latent combos where the .jsonl already exists.")
    parser.add_argument("--combine_features", action="store_true",
                        help="Steer with a single combined vector (weighted sum of top-N decoder "
                             "directions) instead of iterating over latents individually. "
                             "Results go to combined_top{N}/ instead of latent_*/ dirs.")
    parser.add_argument("--combine_weights", default=None,
                        help="Comma-separated weights for --combine_features (signed floats). "
                             "Defaults to the s_visual separation scores from top_features.json, "
                             "or uniform if scores are unavailable. Length must match --n_top_features.")
    parser.add_argument("--feature_source", default=None,
                        help="Label for where the latents came from (e.g. 'vqav2', 'vab'). "
                             "If omitted, inferred from --feature_search_dir or --latent_idx.")
    parser.add_argument("--vilp_mode", default="without_fact",
                        choices=["without_fact", "with_fact"],
                        help="ViLP prompt mode (default: without_fact)")
    parser.add_argument("--vilp_images", default="cf_only",
                        choices=["all", "cf_only"],
                        help="ViLP image subset: cf_only=images 2+3 (default), all=all 3")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_slug = args.model.replace("/", "_")
    base_output_dir = os.path.join(repo_root, args.output_dir, model_slug,
                                   args.dataset, f"layer_{args.target_layer}")
    os.makedirs(base_output_dir, exist_ok=True)

    alphas = [float(a) for a in args.alpha_sweep.split(",")]

    # Resolve feature_source label
    if args.feature_source is not None:
        feature_source = args.feature_source
    elif args.feature_search_dir is not None:
        # Infer from the directory path — take the first path component that isn't
        # a layer/model slug, i.e. the dataset folder name.
        parts = os.path.normpath(args.feature_search_dir).split(os.sep)
        # Walk backwards: skip layer_* and model-slug components
        feature_source = next(
            (p for p in reversed(parts) if not p.startswith("layer_") and p not in ("feature_search",)),
            args.feature_search_dir,
        )
    else:
        feature_source = "manual"

    logger.info("Feature source: %s", feature_source)

    # Determine which latents to steer
    top_features_meta: list[dict] = []   # carries s_visual scores for default weights
    if args.latent_idx is not None:
        latent_indices = [args.latent_idx]
    elif args.feature_search_dir:
        with open(os.path.join(args.feature_search_dir, "top_features.json")) as f:
            top = json.load(f)
        if "top_visual" not in top:
            parser.error(f"top_features.json missing 'top_visual' key. Keys found: {list(top.keys())}")
        top_features_meta = top["top_visual"][:args.n_top_features]
        latent_indices = [f["latent_idx"] for f in top_features_meta]
    else:
        parser.error("Either --latent_idx or --feature_search_dir is required.")

    model, processor = load_gemma3(args.model, device=args.device)
    sae = load_sae(args.target_layer, args.model_size, args.width, args.l0_level, args.device)
    samples = load_dataset(args.dataset, args.n_samples,
                           vilp_mode=args.vilp_mode, vilp_images=args.vilp_images)
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

    if args.combine_features:
        # --- Combined mode: one steering vector from weighted sum of top-N latents ---
        if args.combine_weights is not None:
            weights = [float(w) for w in args.combine_weights.split(",")]
            if len(weights) != len(latent_indices):
                parser.error(f"--combine_weights has {len(weights)} values but {len(latent_indices)} latents selected.")
        elif top_features_meta:
            weights = [f["s_visual"] for f in top_features_meta]
        else:
            weights = None  # uniform

        sv = get_combined_steering_vector(sae, latent_indices, weights)
        label = f"combined_top{len(latent_indices)}"
        run_dir = os.path.join(base_output_dir, label)
        os.makedirs(run_dir, exist_ok=True)

        # Save metadata so the run is reproducible
        meta = {"latent_indices": latent_indices, "weights": weights if weights is not None else [1.0] * len(latent_indices), "feature_source": feature_source}
        with open(os.path.join(run_dir, "combination_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("Combined vector from latents %s with weights %s", latent_indices, meta["weights"])

        run_summaries = []
        for alpha in alphas:
            summary = run_one_latent_alpha(
                model, processor, samples, args.target_layer,
                sv, alpha, run_dir, args.device,
                vanilla_cache=vanilla_cache,
                is_match_fn=is_match_fn,
                skip_existing=args.skip_existing,
            )
            summary["label"] = label
            summary["latent_indices"] = latent_indices
            summary["feature_source"] = feature_source
            run_summaries.append(summary)
            print(f"  {label}  α={alpha:+.0f}  "
                  f"vanilla={summary['vanilla_accuracy']:.3f}  "
                  f"steered={summary['steered_accuracy']:.3f}  "
                  f"Δ={summary['delta']:+.3f}")

        with open(os.path.join(run_dir, "sweep_summary.json"), "w") as f:
            json.dump(run_summaries, f, indent=2)
        all_summaries.extend(run_summaries)

    else:
        # --- Per-latent mode (default, backward compatible) ---
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
                    skip_existing=args.skip_existing,
                )
                summary["latent_idx"] = latent_idx
                summary["feature_source"] = feature_source
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
