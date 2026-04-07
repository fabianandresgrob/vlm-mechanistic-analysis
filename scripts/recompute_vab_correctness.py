"""
Recompute is_correct fields for VAB REVIS/steering results using the proper is_match function.

Run on the server where the `datasets` package is available:
    python scripts/recompute_vab_correctness.py

Rewrites results/revis/vab/*.jsonl and results/steering/*/vab/**/*.jsonl in-place,
backing up originals as *.bak.jsonl.
"""
from __future__ import annotations
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaders import load_vab, get_is_match

is_match = get_is_match("vab")


def build_gt_lookup() -> dict:
    """Load all VAB samples and return id -> ground_truth mapping."""
    print("Loading VAB ground truth...")
    samples = load_vab()
    return {s["id"]: s["answer"] for s in samples}


def recompute_file(path: str, gt: dict) -> tuple[float, float]:
    """Recompute correctness for one JSONL file. Returns (vanilla_acc, steered_acc)."""
    records = [json.loads(l) for l in open(path)]
    n = len(records)
    if n == 0:
        return 0.0, 0.0

    updated = []
    for r in records:
        sid = r.get("sample_id") or r.get("id")
        gt_ans = gt.get(sid, "")
        r["is_correct_vanilla"] = is_match(r.get("vanilla_answer", ""), gt_ans)
        r["is_correct_steered"] = is_match(r.get("steered_answer", ""), gt_ans)
        updated.append(r)

    # Backup and overwrite
    bak = path + ".bak"
    if not os.path.exists(bak):
        os.rename(path, bak)
    with open(path, "w") as f:
        for r in updated:
            f.write(json.dumps(r) + "\n")

    van = sum(r["is_correct_vanilla"] for r in updated) / n
    ste = sum(r["is_correct_steered"] for r in updated) / n
    return van, ste


def recompute_summaries(latent_path: str) -> None:
    """Recompute sweep_summary.json and update all_summaries.json from fixed jsonl files."""
    latent_summaries = []
    for fname in sorted(os.listdir(latent_path)):
        if not fname.startswith("alpha_") or not fname.endswith(".jsonl"):
            continue
        alpha = float(fname[len("alpha_"):-len(".jsonl")])
        records = [json.loads(l) for l in open(os.path.join(latent_path, fname))]
        n = len(records)
        van = sum(r.get("is_correct_vanilla", False) for r in records) / n if n > 0 else 0.0
        ste = sum(r.get("is_correct_steered", False) for r in records) / n if n > 0 else 0.0
        # Preserve existing fields (latent_idx) from old summary if present
        summary_path = os.path.join(latent_path, "sweep_summary.json")
        old_summaries = {}
        if os.path.exists(summary_path):
            for s in json.load(open(summary_path)):
                old_summaries[s.get("alpha")] = s
        entry = old_summaries.get(alpha, {})
        entry.update({"alpha": alpha, "vanilla_accuracy": van, "steered_accuracy": ste,
                       "delta": ste - van, "n": n})
        latent_summaries.append(entry)

    if latent_summaries:
        with open(os.path.join(latent_path, "sweep_summary.json"), "w") as f:
            json.dump(latent_summaries, f, indent=2)


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gt = build_gt_lookup()
    print(f"Loaded {len(gt)} ground truth answers.\n")

    # REVIS results
    revis_vab = os.path.join(repo_root, "results", "revis", "vab")
    if os.path.isdir(revis_vab):
        print("=== REVIS VAB ===")
        for fname in sorted(os.listdir(revis_vab)):
            if not fname.endswith(".jsonl"):
                continue
            path = os.path.join(revis_vab, fname)
            van, ste = recompute_file(path, gt)
            print(f"  {fname}: vanilla={van:.3f}  steered={ste:.3f}  Δ={ste-van:+.3f}")

    # SAE steering results
    steering_root = os.path.join(repo_root, "results", "steering")
    all_summaries_by_layer: dict[str, list] = {}
    for model_dir in os.listdir(steering_root):
        vab_dir = os.path.join(steering_root, model_dir, "vab")
        if not os.path.isdir(vab_dir):
            continue
        for layer_dir in os.listdir(vab_dir):
            layer_path = os.path.join(vab_dir, layer_dir)
            if not os.path.isdir(layer_path):
                continue
            layer_all: list = []
            for latent_dir in os.listdir(layer_path):
                latent_path = os.path.join(layer_path, latent_dir)
                if not os.path.isdir(latent_path):
                    continue
                print(f"\n=== SAE steering {model_dir}/vab/{layer_dir}/{latent_dir} ===")
                for fname in sorted(os.listdir(latent_path)):
                    if not fname.endswith(".jsonl"):
                        continue
                    path = os.path.join(latent_path, fname)
                    van, ste = recompute_file(path, gt)
                    print(f"  {fname}: vanilla={van:.3f}  steered={ste:.3f}  Δ={ste-van:+.3f}")
                recompute_summaries(latent_path)
                sweep = json.load(open(os.path.join(latent_path, "sweep_summary.json")))
                layer_all.extend(sweep)

            # Rewrite all_summaries.json for this layer
            all_path = os.path.join(layer_path, "all_summaries.json")
            if layer_all:
                with open(all_path, "w") as f:
                    json.dump(layer_all, f, indent=2)
                print(f"\nUpdated {all_path}")


if __name__ == "__main__":
    main()
