"""Debug script: inspect pred_vis vs pred_blind for VAB samples.

Runs greedy decoding for blind (zeroed image) and visual conditions on a
small sample of VAB, printing predictions alongside ground truth to diagnose
why n_dvt=0.

Usage:
    python scripts/debug_vab_predictions.py \
        --n_samples 30 \
        --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chain_of_embedding.contrastive_forward import _build_inputs, _greedy_decode, is_vision_dependent
from chain_of_embedding.models.gemma3 import load_gemma3
from data_loaders import get_is_match, load_vab, to_contrastive_sample


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_samples", type=int, default=30)
    p.add_argument("--device", default="cuda")
    p.add_argument("--model", default="google/gemma-3-4b-it")
    p.add_argument("--max_new_tokens", type=int, default=5,
                   help="Slightly higher than production to catch longer outputs")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading {args.model}…")
    model, processor = load_gemma3(args.model, device=args.device)
    is_match_fn = get_is_match("vlms_are_biased")

    print(f"Loading {args.n_samples} VAB samples…")
    raw = load_vab(n_samples=args.n_samples)
    samples = [to_contrastive_sample(d) for d in raw]

    n_changed = 0
    n_changed_correct = 0
    n_same_correct = 0
    n_same_wrong = 0

    print(f"\n{'ID':<6} {'Topic':<22} {'GT':<10} {'pred_vis':<12} {'pred_blind':<12} {'changed':<8} {'vis_ok'}")
    print("-" * 90)

    for raw_s, sample in zip(raw, samples):
        inputs_vis = _build_inputs(sample, processor, args.device)

        # Blind: zero out pixel values
        inputs_blind = dict(inputs_vis)
        inputs_blind["pixel_values"] = torch.zeros_like(inputs_vis["pixel_values"])

        pred_vis   = _greedy_decode(model, inputs_vis,   processor, max_new_tokens=args.max_new_tokens)
        pred_blind = _greedy_decode(model, inputs_blind, processor, max_new_tokens=args.max_new_tokens)

        gt = raw_s.get("answer", "")
        changed = pred_vis.lower().strip() != pred_blind.lower().strip()
        vis_ok  = is_match_fn(pred_vis, gt)

        if changed and vis_ok:
            n_changed_correct += 1
        elif changed:
            n_changed += 1
        elif vis_ok:
            n_same_correct += 1
        else:
            n_same_wrong += 1

        topic = raw_s.get("topic", "")[:20]
        flag = " <-- D_VT" if (changed and vis_ok) else (" <-- changed" if changed else "")
        print(f"{str(raw_s.get('id','')):<6} {topic:<22} {gt:<10} {pred_vis:<12} {pred_blind:<12} {str(changed):<8} {str(vis_ok)}{flag}")

    print("\n" + "=" * 90)
    print(f"Breakdown over {len(samples)} samples:")
    print(f"  changed AND vis_correct  (D_VT)  : {n_changed_correct}")
    print(f"  changed AND vis_wrong            : {n_changed}")
    print(f"  same    AND vis_correct          : {n_same_correct}  ← right for wrong reasons (LP)")
    print(f"  same    AND vis_wrong            : {n_same_wrong}")
    print()
    if n_same_correct > 0:
        print("  NOTE: 'same AND vis_correct' = model answers correctly WITHOUT looking at image.")
        print("        This is the language prior operating correctly — strongest evidence for LP.")
    if n_changed == 0 and n_changed_correct == 0:
        print("  NOTE: answer NEVER changes between blank and real image.")
        print("        The model's output is completely invariant to image content on VAB.")


if __name__ == "__main__":
    main()
