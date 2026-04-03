"""
Exp 3.3: Causal Steering Intervention.

Amplifies or suppresses a visual reliance latent during inference by adding
α × d_j to the residual stream at the target layer, where d_j is the j-th
row of the SAE decoder matrix W_dec.

Intervention:   x_new = x + α * d_j
  α > 0:  amplify visual reliance (push model toward using vision)
  α < 0:  suppress visual reliance (push model toward language prior)

Calibration: Ferrando et al. use α ≈ 2 × ||residual_stream||.
For Gemma 3 4b, residual stream norm is empirically ~100-200, so
α sweep range is [-500, -200, -100, 0, 100, 200, 500].

Implementation: we register a forward hook that fires at the target layer
and adds the steering vector. This avoids re-implementing the full forward
pass.

Reference: Ferrando et al. (ICLR 2025) Section 4.3.
"""
from __future__ import annotations
import logging
from contextlib import contextmanager
import torch
from sae_lens import SAE

logger = logging.getLogger(__name__)


def get_steering_vector(sae: SAE, latent_idx: int) -> torch.Tensor:
    """Return the decoder direction for a given latent.

    This is the j-th row of W_dec (W_dec has shape d_sae × d_in,
    so row j is the direction in d_in space for latent j).

    Args:
        sae: Loaded SAELens SAE.
        latent_idx: Index of the latent to steer.

    Returns:
        Tensor of shape (d_in,), on the same device as the SAE.
    """
    return sae.W_dec[latent_idx].detach()   # (d_in,)


@contextmanager
def steering_hook(
    model,
    target_layer: int,
    steering_vector: torch.Tensor,
    alpha: float,
):
    """Context manager that registers a forward hook to steer at target_layer.

    The hook fires on the output of decoder layer `target_layer` and adds
    alpha * steering_vector to every token position in the hidden states.

    Usage:
        with steering_hook(model, target_layer=20, steering_vector=d_j, alpha=200.0):
            output = model.generate(...)

    Args:
        model:            Gemma3ForConditionalGeneration.
        target_layer:     Layer index (0-based) to hook into.
        steering_vector:  Shape (d_in,). Added to hidden state.
        alpha:            Scaling factor.
    """
    # Gemma3ForConditionalGeneration layer path varies by transformers version:
    #   model.language_model.model.layers  (some versions)
    #   model.model.language_model.layers  (other versions)
    #   model.model.layers                 (plain causal LM)
    if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        layer_module = model.language_model.model.layers[target_layer]
    elif hasattr(model, "model") and hasattr(model.model, "language_model"):
        layer_module = model.model.language_model.layers[target_layer]
    else:
        layer_module = model.model.layers[target_layer]
    sv = steering_vector * alpha  # move device resolution into hook_fn

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            _sv = sv.to(hidden_states.device)
            hidden_states = hidden_states + _sv.unsqueeze(0).unsqueeze(0)
            return (hidden_states,) + output[1:]
        else:
            _sv = sv.to(output.device)
            return output + _sv.unsqueeze(0).unsqueeze(0)

    handle = layer_module.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def steered_generate(
    model,
    processor,
    sample: dict,
    target_layer: int,
    steering_vector: torch.Tensor,
    alpha: float,
    device: str = "cuda",
    max_new_tokens: int = 10,
    precomputed_vanilla: str | None = None,
) -> dict:
    """Generate an answer with and without steering for a single sample.

    Args:
        model:            Loaded Gemma 3 model.
        processor:        Corresponding AutoProcessor.
        sample:           Dict with 'image' (PIL), 'messages', optionally 'answer'.
        target_layer:     Layer to apply steering at.
        steering_vector:  Decoder row d_j, shape (d_in,).
        alpha:            Steering strength. Positive = amplify visual reliance.
        device:           Device string.
        max_new_tokens:   Tokens to generate.

    Returns:
        dict with 'vanilla_answer', 'steered_answer', 'alpha',
        and optionally 'is_correct_vanilla', 'is_correct_steered'.
    """
    text = processor.apply_chat_template(
        sample["messages"], add_generation_prompt=True, tokenize=False
    )
    inputs = processor(text=text, images=[sample["image"]], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Vanilla (skip if precomputed)
    if precomputed_vanilla is not None:
        vanilla_answer = precomputed_vanilla
    else:
        vanilla_out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        vanilla_answer = processor.tokenizer.decode(
            vanilla_out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

    # Steered
    with steering_hook(model, target_layer, steering_vector, alpha):
        steered_out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    steered_answer = processor.tokenizer.decode(
        steered_out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

    result = {
        "vanilla_answer": vanilla_answer,
        "steered_answer": steered_answer,
        "alpha": alpha,
        "target_layer": target_layer,
    }
    if "answer" in sample and sample["answer"]:
        gt = sample["answer"].strip().lower()
        result["is_correct_vanilla"] = vanilla_answer.lower() == gt
        result["is_correct_steered"] = steered_answer.lower() == gt

    return result
