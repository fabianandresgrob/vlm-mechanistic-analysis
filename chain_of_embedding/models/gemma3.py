"""
Gemma 3 multimodal model loader with hook-friendly hidden state extraction.

Gemma 3 IT models are loaded as Gemma3ForConditionalGeneration.
The LLM backbone is at model.language_model; vision encoder at model.vision_tower.
Hidden states are extracted by passing output_hidden_states=True to the forward call.
"""

from __future__ import annotations

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from typing import Optional


def load_gemma3(
    model_id: str = "google/gemma-3-4b-it",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "eager",
) -> tuple[Gemma3ForConditionalGeneration, AutoProcessor]:
    """Load Gemma 3 multimodal model and processor.

    Args:
        model_id: HuggingFace model ID.
        device: Device to load to ("cuda", "cpu", or "auto" for multi-GPU).
        dtype: Model dtype. bfloat16 recommended for most GPUs.
        attn_implementation: "eager", "sdpa", or "flash_attention_2".

    Returns:
        (model, processor) tuple.
    """
    processor = AutoProcessor.from_pretrained(model_id)

    load_kwargs: dict = dict(
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
    )
    if device == "auto":
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = device

    model = Gemma3ForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
    model.eval()
    return model, processor


def num_llm_layers(model: Gemma3ForConditionalGeneration) -> int:
    """Return the number of LLM decoder layers."""
    # model.language_model is the Gemma3TextModel; its config has num_hidden_layers.
    # Accessing via language_model avoids relying on text_config typing on the outer config.
    if hasattr(model, "language_model"):
        return model.language_model.config.num_hidden_layers
    return model.config.text_config.num_hidden_layers  # type: ignore[attr-defined]


def get_lm_head(model: Gemma3ForConditionalGeneration):
    """Return the language model head (unembedding matrix).

    Gemma3ForConditionalGeneration exposes lm_head directly (unlike PaliGemma
    which wraps it under .language_model).
    """
    if hasattr(model, "lm_head"):
        return model.lm_head
    # Fallback for any wrapper variants
    return model.get_output_embeddings()


def get_final_norm(model: Gemma3ForConditionalGeneration):
    """Return the final layer norm of the LLM.

    The text backbone is at model.model (a Gemma3TextModel), with the
    final RMS norm at model.model.norm.
    """
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    # Fallback: walk named modules to find the last LayerNorm/RMSNorm
    norms = [(n, m) for n, m in model.named_modules()
             if "norm" in n.lower() and not any(f"layers.{i}" in n for i in range(200))]
    if norms:
        return norms[-1][1]
    raise AttributeError(f"Cannot find final norm in {type(model).__name__}. "
                         f"Top-level children: {[n for n, _ in model.named_children()]}")


@torch.no_grad()
def forward_with_hidden_states(
    model: Gemma3ForConditionalGeneration,
    inputs: dict,
    include_image: bool = True,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Run a forward pass and return logits + all layer hidden states.

    Args:
        model: Loaded Gemma 3 model.
        inputs: Processed inputs from the processor (on the correct device).
        include_image: If False, zero out pixel_values to simulate text-only pass.

    Returns:
        (logits, hidden_states) where hidden_states is a list of length
        num_layers + 1 (embedding layer + each decoder layer), each tensor
        of shape (batch, seq_len, hidden_dim).
    """
    if not include_image and "pixel_values" in inputs:
        inputs = dict(inputs)  # shallow copy — don't modify caller's dict
        inputs["pixel_values"] = torch.zeros_like(inputs["pixel_values"])

    outputs = model(
        **inputs,
        output_hidden_states=True,
        return_dict=True,
    )
    # outputs.hidden_states: tuple of (num_layers+1) tensors
    hidden_states = list(outputs.hidden_states)
    logits = outputs.logits
    return logits, hidden_states


@torch.no_grad()
def early_exit_logits(
    model: Gemma3ForConditionalGeneration,
    hidden_state: torch.Tensor,
) -> torch.Tensor:
    """Compute logits from an intermediate hidden state via the shared lm_head.

    Args:
        model: Loaded Gemma 3 model.
        hidden_state: Tensor of shape (..., hidden_dim).

    Returns:
        Logits of shape (..., vocab_size).
    """
    norm = get_final_norm(model)
    lm_head = get_lm_head(model)
    return lm_head(norm(hidden_state))


def get_visual_token_mask(
    inputs: dict,
    processor: AutoProcessor,
) -> Optional[torch.Tensor]:
    """Return a boolean mask of shape (batch, seq_len) indicating visual token positions.

    Gemma 3 uses a special image token in the input_ids.
    The processor inserts <image_soft_token> (or equivalent) for each visual patch.

    Returns None if no image tokens are present.
    """
    image_token_id = processor.tokenizer.convert_tokens_to_ids(
        processor.image_token if hasattr(processor, "image_token") else "<image_soft_token>"
    )
    if image_token_id is None:
        return None
    input_ids = inputs.get("input_ids")
    if input_ids is None:
        return None
    return input_ids == image_token_id
