"""p(True) self-evaluation confidence for Medical VQA.

Method:
1. Obtain or receive a proposed answer for an image/question pair.
2. Ask the model whether that proposed answer is correct.
3. Read the next-token probability mass assigned to True vs False and use
   P(True) as confidence. The uncertainty score is 1 - P(True).

Convention:
- ``confidence = P(True)``
- ``uncertainty = 1 - P(True)``
- ``is_correct=True`` means the proposed answer matches the ground truth under
  the same closed/open matching rule used by the repo's sampling confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

from medvlm.configs import ModelConfig, ModelFamily
from medvlm.confidence.sampling import (
    BASE_SAMPLING_PROMPT,
    check_answer_correct,
    normalize_answer,
    parse_answer_text,
)


PTRUE_PROMPT = (
    "Question: {question}\n"
    "Proposed answer: {answer}\n"
    "Is the proposed answer correct? Answer True or False: "
)


@dataclass
class PTrueResult:
    """p(True) result for one Medical VQA sample."""

    question: str
    ground_truth: Optional[str]
    proposed_answer: str
    confidence: float
    uncertainty: float
    p_true: float
    p_false: float
    p_true_raw: float
    p_false_raw: float
    is_correct: Optional[bool]
    answer_type: str
    self_eval_prompt: str
    raw_answer: Optional[str] = None


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


def _format_vlm_prompt(
    processor,
    model_config: ModelConfig,
    image: Any,
    prompt_text: str,
    *,
    add_generation_prompt: bool = True,
) -> Dict[str, Any]:
    """Format an image-text prompt for the supported VLM families."""
    if model_config.model_family == ModelFamily.QWEN_VL:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ]}]
        return processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
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
            add_generation_prompt=add_generation_prompt,
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
            add_generation_prompt=add_generation_prompt,
        )
        return processor(text=text, images=image, return_tensors="pt")

    raise ValueError(f"Unsupported model family: {model_config.model_family}")


def _first_token_ids(tokenizer, candidates: Sequence[str]) -> List[int]:
    """Return candidate first-token ids, avoiding standalone leading spaces."""
    ids = set()
    for candidate in candidates:
        token_ids = tokenizer.encode(candidate, add_special_tokens=False)
        if not token_ids:
            continue
        if candidate.startswith((" ", "\n", "\t")) and len(token_ids) > 1:
            # If the tokenizer splits a leading whitespace into its own token,
            # that first token is not informative for True-vs-False.
            continue
        ids.add(int(token_ids[0]))
    return sorted(ids)


def true_false_token_ids(tokenizer) -> Tuple[List[int], List[int]]:
    """Get token-id sets used for the True/False next-token comparison."""
    true_ids = _first_token_ids(
        tokenizer,
        ("True", "true", "TRUE", " True", "\nTrue"),
    )
    false_ids = _first_token_ids(
        tokenizer,
        ("False", "false", "FALSE", " False", "\nFalse"),
    )
    if not true_ids or not false_ids:
        raise ValueError(
            "Could not resolve True/False token ids for this tokenizer. "
            "Try inspecting tokenizer.encode('True') and tokenizer.encode('False')."
        )
    return true_ids, false_ids


def ptrue_from_logits(
    logits: torch.Tensor,
    tokenizer,
) -> Tuple[float, float, float, float]:
    """Compute normalized P(True) and P(False) from next-token logits.

    Returns:
        ``(p_true, p_false, p_true_raw, p_false_raw)`` where raw probabilities
        are before normalization over the True/False candidate set.
    """
    if logits.ndim == 2:
        next_logits = logits[0]
    elif logits.ndim == 1:
        next_logits = logits
    else:
        raise ValueError("logits must have shape [vocab] or [batch, vocab]")

    true_ids, false_ids = true_false_token_ids(tokenizer)
    probs = torch.softmax(next_logits.float(), dim=-1)
    p_true_raw = probs[true_ids].sum()
    p_false_raw = probs[false_ids].sum()
    denom = p_true_raw + p_false_raw
    if float(denom) <= 0:
        p_true = torch.tensor(0.5, device=probs.device)
        p_false = torch.tensor(0.5, device=probs.device)
    else:
        p_true = p_true_raw / denom
        p_false = p_false_raw / denom
    return (
        float(p_true.detach().cpu()),
        float(p_false.detach().cpu()),
        float(p_true_raw.detach().cpu()),
        float(p_false_raw.detach().cpu()),
    )


def compute_ptrue_for_prompt(
    model,
    processor,
    model_config: ModelConfig,
    image: Any,
    prompt_text: str,
) -> Tuple[float, float, float, float]:
    """Run one self-evaluation forward pass and return True/False probabilities."""
    device = _model_device(model)
    inputs = _format_vlm_prompt(processor, model_config, image, prompt_text)
    inputs = _move_to_device(inputs, device)

    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[:, -1, :]

    return ptrue_from_logits(next_token_logits, processor.tokenizer)


def generate_proposed_answer(
    model,
    processor,
    model_config: ModelConfig,
    sample: Mapping[str, Any],
    *,
    question_type: str = "closed",
    prompt_template: str = BASE_SAMPLING_PROMPT,
    max_new_tokens: int = 64,
) -> Tuple[str, str]:
    """Greedily generate a proposed answer and return normalized/raw answer."""
    device = _model_device(model)
    prompt_text = prompt_template.format(question=sample["question"])
    inputs = _format_vlm_prompt(processor, model_config, sample["image"], prompt_text)
    inputs = _move_to_device(inputs, device)

    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    raw = processor.tokenizer.decode(
        output_ids[0, prompt_len:],
        skip_special_tokens=True,
    )
    parsed = parse_answer_text(raw)
    normalized = normalize_answer(parsed, question_type)
    if normalized is None:
        normalized = parsed.strip().lower() if parsed else raw.strip().lower()
    if not normalized:
        normalized = "unknown"
    return normalized, raw


def compute_ptrue_confidence(
    model,
    processor,
    model_config: ModelConfig,
    examples: Iterable[Mapping[str, Any]],
    *,
    question_type: str = "closed",
    proposed_answers: Optional[Sequence[str]] = None,
    ptrue_prompt: str = PTRUE_PROMPT,
    generate_if_missing: bool = True,
    answer_prompt_template: str = BASE_SAMPLING_PROMPT,
    max_answer_tokens: int = 64,
    show_progress: bool = True,
) -> List[PTrueResult]:
    """Compute p(True) confidence for a sequence of VQA examples.

    Args:
        model, processor, model_config: Loaded VLM triplet.
        examples: Iterable of dicts with ``image``, ``question``, optional
            ``answer`` and optional ``answer_type``.
        question_type: ``"closed"`` or ``"open"`` matching/evaluation rule.
        proposed_answers: Optional externally generated answers. If omitted,
            greedy answers are generated when ``generate_if_missing=True``.
        ptrue_prompt: Prompt template with ``{question}`` and ``{answer}``.
        generate_if_missing: Generate proposed answers when not provided.
        answer_prompt_template: Prompt used for greedy answer generation.
        max_answer_tokens: Max tokens for greedy answer generation.
        show_progress: Show a tqdm bar.

    Returns:
        List of ``PTrueResult``.
    """
    examples_list = list(examples)
    if proposed_answers is not None and len(proposed_answers) != len(examples_list):
        raise ValueError("proposed_answers must have the same length as examples")
    if proposed_answers is None and not generate_if_missing:
        raise ValueError("proposed_answers are required when generate_if_missing=False")

    iterator = enumerate(examples_list)
    if show_progress:
        iterator = tqdm(iterator, total=len(examples_list), desc="p(True)")

    results: List[PTrueResult] = []
    for idx, sample in iterator:
        raw_answer = None
        if proposed_answers is not None:
            proposed = str(proposed_answers[idx]).strip().lower()
        else:
            proposed, raw_answer = generate_proposed_answer(
                model,
                processor,
                model_config,
                sample,
                question_type=question_type,
                prompt_template=answer_prompt_template,
                max_new_tokens=max_answer_tokens,
            )

        prompt_text = ptrue_prompt.format(
            question=sample["question"],
            answer=proposed,
        )
        p_true, p_false, p_true_raw, p_false_raw = compute_ptrue_for_prompt(
            model,
            processor,
            model_config,
            sample["image"],
            prompt_text,
        )

        ground_truth = sample.get("answer")
        gt_norm = str(ground_truth).strip().lower() if ground_truth is not None else None
        is_correct = (
            check_answer_correct(proposed, gt_norm, question_type)
            if gt_norm is not None
            else None
        )

        results.append(PTrueResult(
            question=sample["question"],
            ground_truth=gt_norm,
            proposed_answer=proposed,
            confidence=p_true,
            uncertainty=1.0 - p_true,
            p_true=p_true,
            p_false=p_false,
            p_true_raw=p_true_raw,
            p_false_raw=p_false_raw,
            is_correct=is_correct,
            answer_type=str(sample.get("answer_type", question_type)).lower(),
            self_eval_prompt=prompt_text,
            raw_answer=raw_answer,
        ))

    return results


def _filter_by_answer_type(examples: Iterable[Mapping[str, Any]], answer_type: str) -> List[Mapping[str, Any]]:
    return [
        sample for sample in examples
        if str(sample.get("answer_type", answer_type)).lower() == answer_type
    ]


def compute_ptrue_confidence_by_type(
    model,
    processor,
    model_config: ModelConfig,
    examples: Iterable[Mapping[str, Any]],
    *,
    question_type: str = "all",
    proposed_answers: Optional[Sequence[str]] = None,
    show_progress: bool = True,
    **kwargs,
) -> List[PTrueResult]:
    """Compute p(True) results and report closed/open separately when requested.

    If ``question_type`` is ``"all"``, examples must carry an ``answer_type``
    field and are split into closed/open groups so each group uses the correct
    matching rule. External ``proposed_answers`` are supported in the same order
    as ``examples``.
    """
    examples_list = list(examples)
    if question_type != "all":
        return compute_ptrue_confidence(
            model,
            processor,
            model_config,
            examples_list,
            question_type=question_type,
            proposed_answers=proposed_answers,
            show_progress=show_progress,
            **kwargs,
        )

    if proposed_answers is not None and len(proposed_answers) != len(examples_list):
        raise ValueError("proposed_answers must have the same length as examples")

    results: List[PTrueResult] = []
    for qtype in ("closed", "open"):
        selected = [
            (i, sample) for i, sample in enumerate(examples_list)
            if str(sample.get("answer_type", "")).lower() == qtype
        ]
        if not selected:
            continue
        idxs, subset = zip(*selected)
        subset_answers = (
            [proposed_answers[i] for i in idxs]
            if proposed_answers is not None
            else None
        )
        results.extend(compute_ptrue_confidence(
            model,
            processor,
            model_config,
            subset,
            question_type=qtype,
            proposed_answers=subset_answers,
            show_progress=show_progress,
            **kwargs,
        ))
    return results


__all__ = [
    "PTRUE_PROMPT",
    "PTrueResult",
    "true_false_token_ids",
    "ptrue_from_logits",
    "compute_ptrue_for_prompt",
    "generate_proposed_answer",
    "compute_ptrue_confidence",
    "compute_ptrue_confidence_by_type",
]
