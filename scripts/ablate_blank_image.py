"""Ablation: compare blank-image conditions for the blind forward pass.

Four conditions for the "no visual information" baseline:
  vis        — real image (reference)
  black      — zeroed pixel_values (current approach: black image through SigLIP)
  white      — ones pixel_values (white image through SigLIP)
  text_only  — image tokens stripped from prompt; no pixel_values at all

Metrics reported per condition (vs. vis):
  - agreement rate: fraction of samples where prediction matches vis
  - D_VT rate: fraction where vis≠condition AND vis is correct

Cross-condition agreement (black/white/text_only pairwise) is also printed.

Usage:
    python scripts/ablate_blank_image.py \\
        --n_per_topic 5 \\
        --model google/gemma-3-4b-it \\
        --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chain_of_embedding.contrastive_forward import _build_inputs, _greedy_decode
from chain_of_embedding.models.gemma3 import load_gemma3
from data_loaders import get_is_match, load_vab, to_contrastive_sample

BLANK_CONDITIONS = ("black", "white", "text_only")


def _build_black_inputs(inputs_vis: dict) -> dict:
    return {**inputs_vis, "pixel_values": torch.zeros_like(inputs_vis["pixel_values"])}


def _build_white_inputs(inputs_vis: dict) -> dict:
    return {**inputs_vis, "pixel_values": torch.ones_like(inputs_vis["pixel_values"])}


def _build_text_only_inputs(sample, processor, device: str) -> dict:
    """Strip image content items from messages; tokenize text only."""
    text_only_messages = []
    for msg in sample.messages:
        content = msg["content"]
        if isinstance(content, list):
            content = [c for c in content if c.get("type") != "image"]
        text_only_messages.append({"role": msg["role"], "content": content})
    text = processor.apply_chat_template(
        text_only_messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = processor(text=text, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


def _stratified_sample(raw: list[dict], n_per_topic: int) -> list[dict]:
    buckets: dict[str, list] = defaultdict(list)
    for s in raw:
        buckets[s.get("topic", "unknown")].append(s)
    result = []
    for topic, items in sorted(buckets.items()):
        result.extend(items[:n_per_topic])
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_per_topic", type=int, default=None,
                   help="Samples per VAB topic (stratified). Omit to use all 240 samples.")
    p.add_argument("--model", default="google/gemma-3-4b-it")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_new_tokens", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading {args.model}…")
    model, processor = load_gemma3(args.model, device=args.device)
    is_match_fn = get_is_match("vlms_are_biased")

    print("Loading VAB (resolution=1152)…")
    raw_all = load_vab()  # defaults to resolution=1152
    raw = _stratified_sample(raw_all, args.n_per_topic) if args.n_per_topic else raw_all
    samples = [to_contrastive_sample(d) for d in raw]
    n_topics = len(set(s.get("topic", "unknown") for s in raw))
    print(f"Selected {len(samples)} samples across {n_topics} topics\n")

    # --- per-sample results ---
    # preds[cond] = list of str predictions (one per sample)
    preds: dict[str, list[str]] = {c: [] for c in ("vis", *BLANK_CONDITIONS)}
    gts: list[str] = []

    col_w = 10
    header = (f"{'Topic':<22} {'GT':<6} {'vis':<{col_w}} "
              f"{'black':<{col_w}} {'white':<{col_w}} {'text_only':<{col_w}} flags")
    print(header)
    print("-" * len(header))

    for raw_s, sample in zip(raw, samples):
        gt = raw_s.get("answer", "")
        gts.append(gt)

        inputs_vis = _build_inputs(sample, processor, args.device)

        pred_vis      = _greedy_decode(model, inputs_vis, processor, args.max_new_tokens)
        pred_black    = _greedy_decode(model, _build_black_inputs(inputs_vis), processor, args.max_new_tokens)
        pred_white    = _greedy_decode(model, _build_white_inputs(inputs_vis), processor, args.max_new_tokens)
        pred_text     = _greedy_decode(model, _build_text_only_inputs(sample, processor, args.device), processor, args.max_new_tokens)

        preds["vis"].append(pred_vis)
        preds["black"].append(pred_black)
        preds["white"].append(pred_white)
        preds["text_only"].append(pred_text)

        flags = []
        for cond, pred in (("black", pred_black), ("white", pred_white), ("text", pred_text)):
            if pred_vis.lower().strip() != pred.lower().strip():
                flags.append(f"vis≠{cond}")

        topic = raw_s.get("topic", "unknown")
        print(f"{topic:<22} {gt:<6} {pred_vis:<{col_w}} "
              f"{pred_black:<{col_w}} {pred_white:<{col_w}} {pred_text:<{col_w}} "
              f"{' '.join(flags)}")

    # --- aggregate stats ---
    n = len(gts)
    vis_correct = [is_match_fn(preds["vis"][i], gts[i]) for i in range(n)]

    print("\n" + "=" * 70)
    print(f"\nAgreement with vis  |  D_VT rate  (n={n})")
    print(f"{'Condition':<12}  {'agree_rate':>10}  {'D_VT_rate':>10}")
    print("-" * 38)
    for cond in BLANK_CONDITIONS:
        agree = sum(
            preds["vis"][i].lower().strip() == preds[cond][i].lower().strip()
            for i in range(n)
        )
        dvt = sum(
            preds["vis"][i].lower().strip() != preds[cond][i].lower().strip()
            and vis_correct[i]
            for i in range(n)
        )
        print(f"{cond:<12}  {agree/n:>10.1%}  {dvt/n:>10.1%}")

    print(f"\nCross-condition agreement (blank conditions pairwise):")
    print(f"{'':12}  ", end="")
    for c in BLANK_CONDITIONS:
        print(f"{c:>12}", end="")
    print()
    for c1 in BLANK_CONDITIONS:
        print(f"{c1:<12}  ", end="")
        for c2 in BLANK_CONDITIONS:
            if c1 == c2:
                print(f"{'—':>12}", end="")
            else:
                agree = sum(
                    preds[c1][i].lower().strip() == preds[c2][i].lower().strip()
                    for i in range(n)
                )
                print(f"{agree/n:>12.1%}", end="")
        print()

    print(f"\nvis accuracy: {sum(vis_correct)/n:.1%}")


if __name__ == "__main__":
    main()
