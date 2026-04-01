"""
Exp 2.2: EVA-style corrected decoding.

At the target layer (peak JS divergence from Exp 2.1), the visual signal is
strongest. EVA decoding amplifies it by subtracting the text-only logits from
the multimodal logits at that layer and mixing the correction into the final output.

Formally (following Zhou et al.):
    logits_corrected = logits_final + alpha * (logits_vis_layer - logits_blind_layer)

where logits_vis_layer and logits_blind_layer are early-exit logits at the
target layer for the two conditions.

Novel application: we evaluate this on VLMs Are Biased and ViLP, where nobody
has tested EVA before. Two hypotheses are both publishable:
    H_pos: EVA helps → visual info is recoverable at intermediate layers
    H_neg: EVA doesn't help → the bias is harder than standard hallucination
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F

from chain_of_embedding.models.gemma3 import (
    early_exit_logits,
    forward_with_hidden_states,
)

logger = logging.getLogger(__name__)


@torch.no_grad()
def eva_decode_sample(
    model,
    processor,
    sample: dict,
    target_layer: int,
    alpha: float = 1.0,
    device: str = "cuda",
    max_new_tokens: int = 10,
) -> dict:
    """Run EVA-corrected greedy decoding for a single sample.

    At `target_layer`, computes:
        correction = logits_vis_layer - logits_blind_layer

    Then adds alpha * correction to the final-layer logits before sampling
    the next token. Repeats for max_new_tokens steps.

    Args:
        model:           Loaded Gemma 3 model.
        processor:       Corresponding AutoProcessor.
        sample:          Dict with 'image' (PIL), 'messages', optionally 'answer'.
        target_layer:    Layer index from Exp 2.1 peak JS divergence.
        alpha:           Correction strength. alpha=0 → vanilla greedy.
        device:          Device string.
        max_new_tokens:  Number of tokens to generate.

    Returns:
        dict with keys:
            'vanilla_answer':  str — greedy decode without correction
            'eva_answer':      str — greedy decode with EVA correction
            'is_correct_vanilla': bool (if 'answer' in sample)
            'is_correct_eva':     bool (if 'answer' in sample)
    """
    text = processor.apply_chat_template(
        sample["messages"],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = processor(text=text, images=[sample["image"]], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # --- Vanilla greedy decode ---
    vanilla_out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    vanilla_answer = processor.tokenizer.decode(
        vanilla_out[0, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    # --- EVA corrected decode (token-by-token) ---
    eva_input_ids = inputs["input_ids"].clone()
    eva_attention_mask = inputs["attention_mask"].clone() if "attention_mask" in inputs else None
    eva_generated = []

    for _ in range(max_new_tokens):
        current_inputs = dict(inputs)
        current_inputs["input_ids"] = eva_input_ids
        if eva_attention_mask is not None:
            current_inputs["attention_mask"] = eva_attention_mask

        # Forward pass with image
        logits_final, hs_vis = forward_with_hidden_states(
            model, current_inputs, include_image=True
        )
        # Forward pass without image (same input_ids, zeroed pixel_values)
        _, hs_blind = forward_with_hidden_states(
            model, current_inputs, include_image=False
        )

        # Early-exit logits at target layer (last token position)
        h_vis_layer = hs_vis[target_layer + 1][:, -1, :]    # (1, hidden_dim)
        h_blind_layer = hs_blind[target_layer + 1][:, -1, :]

        logits_vis_layer = early_exit_logits(model, h_vis_layer)    # (1, vocab_size)
        logits_blind_layer = early_exit_logits(model, h_blind_layer)

        # Correction signal
        correction = logits_vis_layer - logits_blind_layer   # (1, vocab_size)

        # Final logits (last token position)
        final_logits = logits_final[:, -1, :]   # (1, vocab_size)

        # Apply EVA correction
        corrected_logits = final_logits + alpha * correction

        # Greedy: argmax
        next_token_id = corrected_logits.argmax(dim=-1)   # (1,)
        eva_generated.append(next_token_id.item())

        # Stop at EOS
        if next_token_id.item() == processor.tokenizer.eos_token_id:
            break

        # Append to sequence — also extend attention_mask to match new length
        eva_input_ids = torch.cat(
            [eva_input_ids, next_token_id.unsqueeze(0)], dim=1
        )
        if eva_attention_mask is not None:
            eva_attention_mask = torch.cat(
                [eva_attention_mask,
                 torch.ones(1, 1, device=eva_attention_mask.device, dtype=eva_attention_mask.dtype)],
                dim=1,
            )

    eva_answer = processor.tokenizer.decode(eva_generated, skip_special_tokens=True).strip()

    result = {
        "vanilla_answer": vanilla_answer,
        "eva_answer": eva_answer,
        "alpha": alpha,
        "target_layer": target_layer,
    }

    if "answer" in sample and sample["answer"]:
        gt = sample["answer"].strip().lower()
        result["is_correct_vanilla"] = vanilla_answer.lower() == gt
        result["is_correct_eva"] = eva_answer.lower() == gt

    return result


@torch.no_grad()
def eva_decode_dataset(
    model,
    processor,
    samples: list[dict],
    target_layer: int,
    alpha: float = 1.0,
    device: str = "cuda",
    max_new_tokens: int = 10,
) -> list[dict]:
    """Run EVA decoding on a list of samples with progress tracking."""
    from tqdm import tqdm

    results = []
    for sample in tqdm(samples, desc=f"EVA decode (layer={target_layer}, α={alpha})"):
        try:
            r = eva_decode_sample(
                model, processor, sample, target_layer, alpha, device, max_new_tokens
            )
            r["id"] = sample.get("id")
            results.append(r)
        except Exception as e:
            logger.warning("Sample %s failed: %s", sample.get("id"), e)
    return results


def accuracy_summary(results: list[dict]) -> dict:
    """Compute accuracy for vanilla and EVA decoding from a results list."""
    vanilla_correct = [r["is_correct_vanilla"] for r in results if "is_correct_vanilla" in r]
    eva_correct = [r["is_correct_eva"] for r in results if "is_correct_eva" in r]

    if not vanilla_correct:
        return {"error": "No correctness labels available"}

    n = len(vanilla_correct)
    return {
        "n": n,
        "vanilla_accuracy": float(sum(vanilla_correct)) / n,
        "eva_accuracy": float(sum(eva_correct)) / n,
        "delta": float(sum(eva_correct) - sum(vanilla_correct)) / n,
    }
