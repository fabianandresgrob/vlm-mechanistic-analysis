"""
Script to run Exp 2.4: Chain-of-Embedding Three-Condition Analysis.

Runs contrastive forward passes (blank / original / counterfactual) on
Gemma 3 and InternVL models, computes per-layer cosine distances, detects
the Visual Integration Point (VIP), and computes Total Visual Integration (TVI).

Usage:
    # Gemma 3 on VLMs Are Biased (full three conditions)
    python scripts/run_chain_of_embedding.py \\
        --model google/gemma-3-4b-it \\
        --dataset vlms_are_biased \\
        --output_dir results/chain_of_embedding/ \\
        --device cuda \\
        --resume

    # Gemma 3 on VQAv2 (two conditions only — no counterfactuals)
    python scripts/run_chain_of_embedding.py \\
        --model google/gemma-3-4b-it \\
        --dataset vqav2 \\
        --n_samples 5000 \\
        --output_dir results/chain_of_embedding/ \\
        --device cuda

Outputs to output_dir/{model_slug}/{dataset}/:
    hidden_states.npz   — hs_blind, hs_vis, hs_cf arrays + metadata
    distances.npz       — per-sample per-layer cosine distances
    tvi.npz             — per-sample TVI values + VIP
    summary.json        — VIP, mean TVI, Spearman ρ, D_VT/D_T stats
    tvi_curve.png       — mean distance curves + VIP marker
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chain_of_embedding.contrastive_forward import (
    ContrastiveSample,
    is_vision_dependent,
    run_contrastive_forward,
)
from chain_of_embedding.models.gemma3 import load_gemma3, num_llm_layers
from chain_of_embedding.tvi import compute_tvi, tvi_statistics
from chain_of_embedding.vip import aggregate_vip, compute_layer_distances, detect_vip

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_vqav2_samples(n_samples: int) -> list[ContrastiveSample]:
    from datasets import load_dataset
    logger.info("Loading VQAv2 validation (n=%d)…", n_samples)
    ds = load_dataset("lmms-lab/VQAv2", split="validation")
    samples = []
    for item in ds.select(range(min(n_samples, len(ds)))):
        samples.append(ContrastiveSample(
            id=item.get("question_id", len(samples)),
            image=item["image"],
            cf_image=None,   # VQAv2 has no counterfactuals
            messages=[{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": item["question"]},
            ]}],
            answer=item.get("multiple_choice_answer", ""),
        ))
    return samples


def load_vlms_are_biased_samples(dataset_id: str, split: str, n_samples: int) -> list[ContrastiveSample]:
    from datasets import load_dataset
    logger.info("Loading VLMs Are Biased from %s / %s…", dataset_id, split)
    ds = load_dataset(dataset_id, split=split)
    samples = []
    for item in ds.select(range(min(n_samples, len(ds)))):
        question = item.get("question") or item.get("query") or ""
        answer = item.get("answer") or item.get("gt_answer") or ""
        # Load counterfactual image — field name varies by dataset version
        cf_image = item.get("cf_image") or item.get("counterfactual_image")
        samples.append(ContrastiveSample(
            id=item.get("id") or item.get("question_id") or len(samples),
            image=item.get("image"),
            cf_image=cf_image,
            messages=[{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ]}],
            answer=answer,
        ))
    logger.info("Loaded %d samples (%d with counterfactuals).",
                len(samples), sum(s.cf_image is not None for s in samples))
    return samples


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_tvi_curve(agg: dict, output_path: str, title: str = "") -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    layers = np.arange(len(agg["mean_d_vis"]))
    vip = agg["vip_median"]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(layers, agg["mean_d_vis"], "o-", label="cos_dist(vis, blind)", color="steelblue")
    if agg.get("mean_d_cf") is not None:
        ax.plot(layers, agg["mean_d_cf"], "s-", label="cos_dist(cf, blind)", color="coral")
    if agg.get("mean_d_disc") is not None:
        ax.plot(layers, agg["mean_d_disc"], "^--", label="cos_dist(vis, cf)", color="seagreen")
    ax.axvline(vip, color="black", linestyle=":", linewidth=1.5, label=f"VIP (median L{vip})")
    ax.set_xlabel("LLM Layer")
    ax.set_ylabel("Mean Cosine Distance")
    ax.set_title(title or "Chain-of-Embedding Distance Curves")
    ax.legend(fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Exp 2.4: Chain-of-Embedding Analysis")
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument(
        "--dataset", default="vlms_are_biased",
        choices=["vqav2", "vlms_are_biased", "vilp", "vlind"],
    )
    parser.add_argument("--vab_dataset_id", default="anvo25/vlms-are-biased")
    parser.add_argument("--vab_split", default="main")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Max samples (default: all)")
    parser.add_argument("--output_dir", default="results/chain_of_embedding/")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no_decode", action="store_true",
                        help="Skip greedy decoding (faster, no D_VT/D_T split)")
    parser.add_argument("--vip_baseline_layers", type=int, default=4)
    parser.add_argument("--vip_threshold_k", type=float, default=2.0)
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_slug = args.model.replace("/", "_")
    output_dir = os.path.join(repo_root, args.output_dir, model_slug, args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    # --- Resume check ---
    summary_path = os.path.join(output_dir, "summary.json")
    if args.resume and os.path.exists(summary_path):
        logger.info("Results already exist at %s — skipping.", output_dir)
        with open(summary_path) as f:
            print(json.load(f))
        return

    # --- Load samples ---
    n = args.n_samples or 999999
    if args.dataset == "vqav2":
        samples = load_vqav2_samples(n)
    elif args.dataset == "vlms_are_biased":
        samples = load_vlms_are_biased_samples(args.vab_dataset_id, args.vab_split, n)
    else:
        raise ValueError(f"Dataset {args.dataset!r} loader not yet implemented.")

    # --- Load model ---
    logger.info("Loading %s…", args.model)
    model, processor = load_gemma3(args.model, device=args.device)
    n_layers = num_llm_layers(model)

    # --- Run contrastive forward passes ---
    from tqdm import tqdm

    all_hs_blind, all_hs_vis, all_hs_cf = [], [], []
    all_distances = []
    all_vip = []
    all_tvi = []
    is_vis_dep = []
    sample_ids = []

    for sample in tqdm(samples, desc="Contrastive forward"):
        result = run_contrastive_forward(
            model, processor, sample,
            device=args.device,
            decode_answers=not args.no_decode,
        )
        if result is None:
            continue

        distances = compute_layer_distances(
            result.hs_blind, result.hs_vis,
            hs_cf=result.hs_cf if result.has_cf else None,
        )

        vip = detect_vip(
            distances["d_vis"],
            d_cf=distances.get("d_cf"),
            d_disc=distances.get("d_disc"),
            baseline_layers=args.vip_baseline_layers,
            threshold_k=args.vip_threshold_k,
        )

        tvi = compute_tvi(result.hs_blind, result.hs_vis, vip, normalize_by_dim=True)

        all_hs_blind.append(result.hs_blind)
        all_hs_vis.append(result.hs_vis)
        if result.has_cf and result.hs_cf is not None:
            all_hs_cf.append(result.hs_cf)
        all_distances.append(distances)
        all_vip.append(vip)
        all_tvi.append(tvi)
        sample_ids.append(result.sample_id)

        vis_dep = is_vision_dependent(result, ground_truth=sample.answer)
        is_vis_dep.append(float(vis_dep))

    logger.info("Processed %d / %d samples.", len(sample_ids), len(samples))

    # --- Aggregate ---
    agg = aggregate_vip(
        all_distances,
        baseline_layers=args.vip_baseline_layers,
        threshold_k=args.vip_threshold_k,
    )

    tvi_arr = np.array(all_tvi)
    is_vis_dep_arr = np.array(is_vis_dep) if not args.no_decode else None
    stats = tvi_statistics(tvi_arr, is_vision_dependent=is_vis_dep_arr)

    # --- Save ---
    np.savez(
        os.path.join(output_dir, "hidden_states.npz"),
        hs_blind=np.stack(all_hs_blind),
        hs_vis=np.stack(all_hs_vis),
        hs_cf=np.stack(all_hs_cf) if all_hs_cf else np.array([]),
        sample_ids=np.array(sample_ids, dtype=object),
    )

    np.savez(
        os.path.join(output_dir, "distances.npz"),
        d_vis=np.stack([d["d_vis"] for d in all_distances]),
        d_cf=np.stack([d["d_cf"] for d in all_distances if d["d_cf"] is not None]) if any(d["d_cf"] is not None for d in all_distances) else np.array([]),
        d_disc=np.stack([d["d_disc"] for d in all_distances if d["d_disc"] is not None]) if any(d["d_disc"] is not None for d in all_distances) else np.array([]),
        sample_ids=np.array(sample_ids, dtype=object),
    )

    np.savez(
        os.path.join(output_dir, "tvi.npz"),
        tvi=tvi_arr,
        vip=np.array(all_vip),
        is_vision_dependent=is_vis_dep_arr if is_vis_dep_arr is not None else np.array([]),
        sample_ids=np.array(sample_ids, dtype=object),
    )

    summary = {
        "model_id": args.model,
        "dataset": args.dataset,
        "n_samples": len(sample_ids),
        "n_layers": n_layers,
        "vip_mean": agg["vip_mean"],
        "vip_median": agg["vip_median"],
        **{k: float(v) if isinstance(v, (float, np.floating)) else v
           for k, v in stats.items()},
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # --- Plot ---
    plot_tvi_curve(
        agg,
        output_path=os.path.join(output_dir, "tvi_curve.png"),
        title=f"{args.model} | {args.dataset} | VIP={agg['vip_median']}",
    )

    print(f"\nVIP (median): layer {agg['vip_median']} (mean: {agg['vip_mean']:.1f})")
    print(f"Mean TVI:     {stats['mean']:.4f} ± {stats['std']:.4f}")
    if "tvi_mean_dvt" in stats:
        print(f"TVI D_VT:     {stats['tvi_mean_dvt']:.4f}  (n={stats['n_dvt']})")
        print(f"TVI D_T:      {stats['tvi_mean_dt']:.4f}  (n={stats['n_dt']})")
    if "spearman_rho" in stats:
        print(f"Spearman ρ:   {stats['spearman_rho']:.3f} (p={stats['spearman_pval']:.2e})")


if __name__ == "__main__":
    main()
