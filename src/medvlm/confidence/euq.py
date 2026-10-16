"""Minimal EUQ helper primitives for answer-conditioned evidence runs.

This module intentionally exposes only the low-level helpers used by
``run_unified_answer_euq.py``. It does not register a public confidence
API and does not implement dataset-level EUQ scoring.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch

from ..configs import ModelConfig, ModelFamily


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _move_to_device(inputs: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in inputs.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def _format_qwen_vl_prompt(processor, messages) -> Dict[str, Any]:
    """Format Qwen-VL messages with qwen_vl_utils when available."""
    try:
        from qwen_vl_utils import process_vision_info
    except Exception:
        return processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    return processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )


def _format_prompt(
    processor,
    model_config: ModelConfig,
    image: Any,
    prompt_text: str,
) -> Dict[str, Any]:
    """Format an image-text prompt for supported VLM families."""
    if model_config.model_family == ModelFamily.QWEN_VL:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ]}]
        return _format_qwen_vl_prompt(processor, messages)

    if model_config.model_family == ModelFamily.GEMMA:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ]}]
        return processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

    if model_config.model_family == ModelFamily.INTERNVL:
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt_text},
        ]}]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return processor(images=image, text=text, return_tensors="pt")

    if model_config.model_family in (ModelFamily.LLAVA, ModelFamily.LLAVA_NEXT):
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt_text},
        ]}]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return processor(text=text, images=image, return_tensors="pt")

    raise ValueError(f"Unsupported model family: {model_config.model_family}")


def get_output_head(model):
    """Return the model output projection module."""
    if hasattr(model, "get_output_embeddings"):
        head = model.get_output_embeddings()
        if head is not None and hasattr(head, "weight"):
            return head

    candidates = [
        "lm_head",
        "language_model.lm_head",
        "model.lm_head",
        "model.language_model.lm_head",
    ]
    for path in candidates:
        obj = model
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok and hasattr(obj, "weight"):
            return obj

    raise ValueError("Could not locate an output projection/lm_head on this model")


def _get_vocab_mean_weight(head, *, chunk_size: int = 8192) -> torch.Tensor:
    """Compute and cache the vocabulary-mean output weight vector."""
    weight = head.weight
    cached = getattr(head, "_medvlm_euq_vocab_mean", None)
    meta = getattr(head, "_medvlm_euq_vocab_mean_meta", None)
    current_meta = (tuple(weight.shape), str(weight.device), str(weight.dtype))
    if cached is not None and meta == current_meta:
        return cached

    vocab_size, hidden_size = weight.shape
    total = torch.zeros(hidden_size, device=weight.device, dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, vocab_size, chunk_size):
            stop = min(start + chunk_size, vocab_size)
            total += weight[start:stop].detach().float().sum(dim=0)
    mean = total / float(vocab_size)

    setattr(head, "_medvlm_euq_vocab_mean", mean)
    setattr(head, "_medvlm_euq_vocab_mean_meta", current_meta)
    return mean


def _local_evidence_scores(
    contribution: torch.Tensor,
    *,
    evidence_scale: Optional[float] = None,
) -> Dict[str, float]:
    """Convert signed feature contributions into local CF/IG scores."""
    scale = float(evidence_scale) if evidence_scale is not None else float(contribution.numel()) ** 0.5
    if scale <= 0:
        scale = 1.0
    contrib = contribution.double() / scale
    e_pos = torch.relu(contrib).sum()
    e_neg = torch.relu(-contrib).sum()

    e_pos_c = torch.clamp(e_pos, max=50.0)
    e_neg_c = torch.clamp(e_neg, max=50.0)
    m_pos = 1.0 - torch.exp(-e_pos_c)
    m_neg = 1.0 - torch.exp(-e_neg_c)
    cf = m_pos * m_neg
    ig = torch.exp(-torch.clamp(e_pos + e_neg, max=50.0))

    return {
        "e_pos": float(e_pos.detach().cpu()),
        "e_neg": float(e_neg.detach().cpu()),
        "cf": float(cf.detach().cpu()),
        "ig": float(ig.detach().cpu()),
    }


__all__ = [
    "_format_prompt",
    "_get_vocab_mean_weight",
    "_local_evidence_scores",
    "_model_device",
    "_move_to_device",
    "get_output_head",
]
