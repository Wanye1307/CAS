"""Teacher-forced answer likelihood and image counterfactuals for CAS-UQ."""


from __future__ import annotations


import json


import math


import sys


from typing import Any, Dict, Mapping, Optional, Sequence


import torch


from PIL import ImageFilter


from confidence.grounding import _model_device, _format_prompt, _move_to_device, _to_rgb_image, make_baseline_image


def extend_sequence_inputs(
    inputs: Mapping[str, Any],
    generated_ids: Sequence[int],
    *,
    prompt_len: int,
    device: torch.device,
) -> Dict[str, Any]:
    gen_len = len(generated_ids)
    full_ids = torch.cat(
        [
            inputs["input_ids"],
            torch.tensor([list(generated_ids)], dtype=inputs["input_ids"].dtype, device=device),
        ],
        dim=1,
    )
    out: Dict[str, Any] = {}
    for key, value in inputs.items():
        if key in {"input_ids", "attention_mask"}:
            continue
        if (
            key == "position_ids"
            and torch.is_tensor(value)
            and value.ndim == 2
            and value.shape[0] == 1
            and value.shape[1] == prompt_len
        ):
            increments = torch.arange(1, gen_len + 1, dtype=value.dtype, device=device).view(1, gen_len)
            out[key] = torch.cat([value.to(device), value[:, -1:].to(device) + increments], dim=1)
        elif (
            torch.is_tensor(value)
            and value.ndim >= 2
            and value.shape[0] == 1
            and value.shape[1] == prompt_len
        ):
            tail = value[:, -1:].to(device).expand(*value.shape[:1], gen_len, *value.shape[2:])
            out[key] = torch.cat([value.to(device), tail], dim=1)
        else:
            out[key] = value
    out["input_ids"] = full_ids
    prompt_mask = inputs.get("attention_mask")
    if torch.is_tensor(prompt_mask) and tuple(prompt_mask.shape) == tuple(inputs["input_ids"].shape):
        tail_mask = torch.ones((prompt_mask.shape[0], gen_len), dtype=prompt_mask.dtype, device=device)
        out["attention_mask"] = torch.cat([prompt_mask.to(device), tail_mask], dim=1)
    else:
        out["attention_mask"] = torch.ones_like(full_ids, device=device)
    if tuple(out["attention_mask"].shape) != tuple(full_ids.shape):
        raise ValueError(
            f"attention_mask shape {tuple(out['attention_mask'].shape)} does not match "
            f"input_ids shape {tuple(full_ids.shape)}"
        )
    position_ids = out.get("position_ids")
    if torch.is_tensor(position_ids) and position_ids.ndim == 2 and position_ids.shape[0] == 1:
        if int(position_ids.shape[1]) != int(full_ids.shape[1]):
            raise ValueError(
                f"position_ids length {int(position_ids.shape[1])} does not match "
                f"input_ids length {int(full_ids.shape[1])}"
            )
    return out


def decode_token(processor: Any, token_id: int) -> str:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return str(int(token_id))
    try:
        return str(tokenizer.decode([int(token_id)]))
    except Exception:
        return str(int(token_id))


def invalid_likelihood(reason: str) -> Dict[str, Any]:
    return {
        "valid": False,
        "invalid_reason": reason,
        "sum_logprob": 0.0,
        "mean_logprob": float("-inf"),
        "confidence": 0.0,
        "nll": float("inf"),
        "token_count": 0,
        "token_logprobs": [],
    }


def answer_token_likelihood(
    model: Any,
    processor: Any,
    model_config: Any,
    image: Optional[Any],
    prompt_text: str,
    generated_ids: Sequence[int],
    *,
    debug_token_alignment: bool = False,
    debug_label: str = "",
    debug_token_limit: int = 8,
    strict_alignment: bool = True,
) -> Dict[str, Any]:
    if not generated_ids:
        return invalid_likelihood("empty_generated_ids")

    device = _model_device(model)
    prompt_inputs = _format_prompt(processor, model_config, image, prompt_text)
    prompt_inputs = _move_to_device(prompt_inputs, device)
    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    forward_inputs = extend_sequence_inputs(
        prompt_inputs,
        generated_ids,
        prompt_len=prompt_len,
        device=device,
    )

    with torch.no_grad():
        outputs = model(**forward_inputs)
        logits = outputs.logits[0].float()

    input_len = int(forward_inputs["input_ids"].shape[1])
    logits_len = int(logits.shape[0])
    if logits_len != input_len:
        message = (
            "Token alignment is unsafe: model logits length does not match teacher-forced input_ids length "
            f"for {debug_label or 'sample'} (logits={logits_len}, input_ids={input_len}, "
            f"prompt_len={prompt_len}, generated_len={len(generated_ids)}). This can happen if the VLM "
            "expands image placeholders inside forward()."
        )
        if strict_alignment:
            raise RuntimeError(message)
        print(f"WARNING: {message}", file=sys.stderr, flush=True)

    values = []
    debug_rows = []
    for i, token_id in enumerate(generated_ids):
        pos = prompt_len + i - 1
        if pos < 0 or pos >= logits.shape[0]:
            continue
        log_probs = torch.log_softmax(logits[pos], dim=-1)
        token_logprob = log_probs[int(token_id)]
        values.append(token_logprob)
        if debug_token_alignment and len(debug_rows) < int(debug_token_limit):
            top_logprob, top_id = torch.max(log_probs, dim=-1)
            debug_rows.append(
                {
                    "token_index": int(i),
                    "logit_pos": int(pos),
                    "target_id": int(token_id),
                    "target_text": decode_token(processor, int(token_id)),
                    "target_logprob": float(token_logprob.detach().cpu()),
                    "top_id": int(top_id.detach().cpu()),
                    "top_text": decode_token(processor, int(top_id.detach().cpu())),
                    "top_logprob": float(top_logprob.detach().cpu()),
                }
            )

    if debug_token_alignment:
        print(
            json.dumps(
                {
                    "event": "token_alignment_debug",
                    "label": debug_label,
                    "prompt_len": prompt_len,
                    "input_len": input_len,
                    "logits_len": logits_len,
                    "generated_len": len(generated_ids),
                    "tokens": debug_rows,
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )

    if not values:
        out = invalid_likelihood("no_aligned_token_positions")
        out.update({"input_token_count": input_len, "logits_token_count": logits_len})
        return out
    stacked = torch.stack(values)
    mean_logprob = float(stacked.mean().detach().cpu())
    return {
        "valid": True,
        "sum_logprob": float(stacked.sum().detach().cpu()),
        "mean_logprob": mean_logprob,
        "confidence": float(math.exp(max(mean_logprob, -80.0))),
        "nll": float(-mean_logprob),
        "token_count": int(stacked.numel()),
        "input_token_count": input_len,
        "logits_token_count": logits_len,
        "token_logprobs": [float(v.detach().cpu()) for v in stacked],
    }


def blur_image(image: Any, radius: float = 8.0):
    return _to_rgb_image(image).filter(ImageFilter.GaussianBlur(radius=radius))


def baseline_image(name: str, image: Any, *, seed: int) -> Optional[Any]:
    if name == "text":
        return None
    if name == "blur":
        return blur_image(image)
    return make_baseline_image(image, baseline=name, seed=seed)
