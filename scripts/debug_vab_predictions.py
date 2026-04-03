"""Debug script: inspect pred_vis vs pred_blind vs pred_text_only for VAB samples.

Three conditions compared:
  - vis:        real image
  - blank:      zeroed pixel_values (black image fed through SigLIP — current blind condition)
  - text_only:  prompt with NO image tokens at all (pure text, no pixel_values)

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


def _build_text_only_inputs(sample, processor, device: str) -> dict:
    """Build inputs with NO image tokens — strip the image content item from messages."""
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
    # Pass no images — text-only tokenization
    inputs = processor(text=text, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_per_topic", type=int, default=5,
                   help="Number of samples to take per topic (stratified)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--model", default="google/gemma-3-4b-it")
    p.add_argument("--max_new_tokens", type=int, default=5)
    return p.parse_args()


def _stratified_sample(raw: list[dict], n_per_topic: int) -> list[dict]:
    """Take up to n_per_topic samples from each unique topic."""
    from collections import defaultdict
    buckets: dict[str, list] = defaultdict(list)
    for s in raw:
        buckets[s.get("topic", "unknown")].append(s)
    result = []
    for topic, items in sorted(buckets.items()):
        result.extend(items[:n_per_topic])
    return result


def main():
    args = parse_args()

    print(f"Loading {args.model}…")
    model, processor = load_gemma3(args.model, device=args.device)
    is_match_fn = get_is_match("vlms_are_biased")

    print(f"Loading VAB (stratified: {args.n_per_topic} per topic)…")
    raw_all = load_vab()
    raw = _stratified_sample(raw_all, args.n_per_topic)
    samples = [to_contrastive_sample(d) for d in raw]
    print(f"Selected {len(samples)} samples across {len(set(s.get('topic') for s in raw))} topics\n")

    from collections import defaultdict
    n_dvt_blank = 0
    n_dvt_text  = 0
    # per-topic: counts of (vis==blank, vis==text, vis_correct)
    topic_stats: dict[str, dict] = defaultdict(lambda: {"n": 0, "vis_eq_blank": 0, "vis_eq_text": 0, "vis_correct": 0, "text_correct": 0})

    print(f"{'Topic':<22} {'GT':<6} {'vis':<8} {'blank':<8} {'text':<8} {'vis_ok':<7} {'flags'}")
    print("-" * 78)

    for raw_s, sample in zip(raw, samples):
        inputs_vis   = _build_inputs(sample, processor, args.device)
        inputs_blank = {**inputs_vis, "pixel_values": torch.zeros_like(inputs_vis["pixel_values"])}
        inputs_text  = _build_text_only_inputs(sample, processor, args.device)

        pred_vis   = _greedy_decode(model, inputs_vis,   processor, args.max_new_tokens)
        pred_blank = _greedy_decode(model, inputs_blank, processor, args.max_new_tokens)
        pred_text  = _greedy_decode(model, inputs_text,  processor, args.max_new_tokens)

        gt     = raw_s.get("answer", "")
        vis_ok = is_match_fn(pred_vis, gt)
        txt_ok = is_match_fn(pred_text, gt)

        changed_blank = pred_vis.lower().strip() != pred_blank.lower().strip()
        changed_text  = pred_vis.lower().strip() != pred_text.lower().strip()

        if changed_blank and vis_ok: n_dvt_blank += 1
        if changed_text  and vis_ok: n_dvt_text  += 1

        topic = raw_s.get("topic", "unknown")
        ts = topic_stats[topic]
        ts["n"]            += 1
        ts["vis_eq_blank"] += int(not changed_blank)
        ts["vis_eq_text"]  += int(not changed_text)
        ts["vis_correct"]  += int(vis_ok)
        ts["text_correct"] += int(txt_ok)

        flags = []
        if changed_blank: flags.append("vis≠blank")
        if changed_text:  flags.append("vis≠text")
        print(f"{topic:<22} {gt:<6} {pred_vis:<8} {pred_blank:<8} {pred_text:<8} {str(vis_ok):<7} {' '.join(flags)}")

    print("\n" + "=" * 78)
    print(f"\nPer-topic summary ({args.n_per_topic} samples each):")
    print(f"  {'Topic':<22} {'n':>3}  {'vis=blank':>9}  {'vis=text':>8}  {'vis_acc':>7}  {'text_acc':>8}")
    print("  " + "-" * 65)
    for topic, ts in sorted(topic_stats.items()):
        n = ts["n"]
        print(f"  {topic:<22} {n:>3}  "
              f"{ts['vis_eq_blank']:>4}/{n} ({100*ts['vis_eq_blank']/n:4.0f}%)  "
              f"{ts['vis_eq_text']:>4}/{n} ({100*ts['vis_eq_text']/n:4.0f}%)  "
              f"{ts['vis_correct']:>4}/{n} ({100*ts['vis_correct']/n:4.0f}%)  "
              f"{ts['text_correct']:>4}/{n} ({100*ts['text_correct']/n:4.0f}%)")

    print(f"\nOverall D_VT (vis≠blank AND vis_correct): {n_dvt_blank}")
    print(f"Overall D_VT (vis≠text  AND vis_correct): {n_dvt_text}")


if __name__ == "__main__":
    main()
