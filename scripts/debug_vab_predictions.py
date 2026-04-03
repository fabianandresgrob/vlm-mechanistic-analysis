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
    p.add_argument("--n_samples", type=int, default=30)
    p.add_argument("--device", default="cuda")
    p.add_argument("--model", default="google/gemma-3-4b-it")
    p.add_argument("--max_new_tokens", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading {args.model}…")
    model, processor = load_gemma3(args.model, device=args.device)
    is_match_fn = get_is_match("vlms_are_biased")

    print(f"Loading {args.n_samples} VAB samples…")
    raw = load_vab(n_samples=args.n_samples)
    samples = [to_contrastive_sample(d) for d in raw]

    counts = {"vis_only": 0, "blank_only": 0, "text_only": 0, "all_same": 0, "mixed": 0}
    n_dvt_blank = 0   # changed vis vs blank AND vis correct (current definition)
    n_dvt_text  = 0   # changed vis vs text-only AND vis correct

    print(f"\n{'ID':<5} {'Topic':<18} {'GT':<6} {'vis':<8} {'blank':<8} {'text':<8} {'match_fn_ok'}")
    print("-" * 70)

    for raw_s, sample in zip(raw, samples):
        inputs_vis = _build_inputs(sample, processor, args.device)

        inputs_blank = dict(inputs_vis)
        inputs_blank["pixel_values"] = torch.zeros_like(inputs_vis["pixel_values"])

        inputs_text = _build_text_only_inputs(sample, processor, args.device)

        pred_vis   = _greedy_decode(model, inputs_vis,   processor, args.max_new_tokens)
        pred_blank = _greedy_decode(model, inputs_blank, processor, args.max_new_tokens)
        pred_text  = _greedy_decode(model, inputs_text,  processor, args.max_new_tokens)

        gt     = raw_s.get("answer", "")
        vis_ok = is_match_fn(pred_vis, gt)

        changed_blank = pred_vis.lower().strip() != pred_blank.lower().strip()
        changed_text  = pred_vis.lower().strip() != pred_text.lower().strip()

        if changed_blank and vis_ok:
            n_dvt_blank += 1
        if changed_text and vis_ok:
            n_dvt_text += 1

        topic = raw_s.get("topic", "")[:16]
        flags = []
        if changed_blank: flags.append("vis≠blank")
        if changed_text:  flags.append("vis≠text")
        print(f"{str(raw_s.get('id','')):<5} {topic:<18} {gt:<6} {pred_vis:<8} {pred_blank:<8} {pred_text:<8} {str(vis_ok):<5}  {' '.join(flags)}")

    print("\n" + "=" * 70)
    print(f"Over {len(samples)} samples:")
    print(f"  D_VT (vis≠blank AND vis_correct): {n_dvt_blank}  ← current chain-of-embedding definition")
    print(f"  D_VT (vis≠text  AND vis_correct): {n_dvt_text}   ← text-only blind condition")
    print()
    print("If vis==blank==text on everything: model output is fully text-determined,")
    print("image tokens (even blank ones) have zero effect on the generated answer.")
    print("If vis==blank but vis≠text: blank image acts like text-only — SigLIP black")
    print("image tokens are being ignored, but prompt structure differs.")


if __name__ == "__main__":
    main()
