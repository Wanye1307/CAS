"""Closed-ended P(yes/no) confidence for Medical VQA.

This is the lightweight yes/no baseline used in our closed-ended experiments.
It performs one forward pass at the answer position, reads the next-token
probability mass assigned to yes/no variants, normalizes within the binary
candidate set, and uses ``max(P(yes), P(no))`` as confidence.

Conventions:
- ``confidence = max(P(yes), P(no))``
- ``uncertainty = 1 - confidence``
- AUROC error detection should use ``uncertainty`` as the score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

from medvlm.configs import ModelConfig, ModelFamily
from medvlm.confidence.sampling import check_answer_correct


PYES_PROMPT = (
    "You are a medical AI assistant. Look at the provided medical image "
    "and answer the following question with yes or no.\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


@dataclass
class PYesResult:
    """P(yes/no) result for one closed-ended VQA sample."""

    question: str
    ground_truth: Optional[str]
    predicted: str
    confidence: float
    uncertainty: float
    p_yes: float
    p_no: float
    p_yes_raw: float
    p_no_raw: float
    is_correct: Optional[bool]
    answer_type: str
    prompt: str
    yes_token_ids: List[int]
    no_token_ids: List[int]


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


def _first_token_ids(tokenizer, candidates: Sequence[str]) -> List[int]:
    """Return candidate first-token ids, avoiding standalone whitespace ids."""
    ids = set()
    for candidate in candidates:
        token_ids = tokenizer.encode(candidate, add_special_tokens=False)
        if not token_ids:
            continue
        if candidate.startswith((" ", "\n", "\t")) and len(token_ids) > 1:
            continue
        ids.add(int(token_ids[0]))
    return sorted(ids)


def yes_no_token_ids(tokenizer) -> Tuple[List[int], List[int]]:
    """Return tokenizer first-token id sets for yes/no comparison."""
    yes_ids = _first_token_ids(
        tokenizer,
        ("yes", "Yes", "YES", " yes", "\nyes"),
    )
    no_ids = _first_token_ids(
        tokenizer,
        ("no", "No", "NO", " no", "\nno"),
    )
    if not yes_ids or not no_ids:
        raise ValueError(
            "Could not resolve yes/no token ids for this tokenizer. "
            "Try inspecting tokenizer.encode('yes') and tokenizer.encode('no')."
        )
    return yes_ids, no_ids


def pyes_from_logits(
    logits: torch.Tensor,
    tokenizer,
) -> Tuple[float, float, float, float]:
    """Compute normalized P(yes) and P(no) from next-token logits.

    Returns:
        ``(p_yes, p_no, p_yes_raw, p_no_raw)`` where raw probabilities are
        measured against the full vocabulary, and normalized probabilities are
        renormalized within the yes/no candidate set.
    """
    if logits.ndim == 2:
        next_logits = logits[0]
    elif logits.ndim == 1:
        next_logits = logits
    else:
        raise ValueError("logits must have shape [vocab] or [batch, vocab]")

    next_logits = next_logits.float()
    yes_ids, no_ids = yes_no_token_ids(tokenizer)
    yes_index = torch.tensor(yes_ids, device=next_logits.device, dtype=torch.long)
    no_index = torch.tensor(no_ids, device=next_logits.device, dtype=torch.long)

    log_yes = torch.logsumexp(next_logits.index_select(0, yes_index), dim=0)
    log_no = torch.logsumexp(next_logits.index_select(0, no_index), dim=0)
    log_total = torch.logsumexp(next_logits, dim=0)
    log_binary = torch.logsumexp(torch.stack([log_yes, log_no]), dim=0)

    p_yes_raw = torch.exp(log_yes - log_total)
    p_no_raw = torch.exp(log_no - log_total)
    p_yes = torch.exp(log_yes - log_binary)
    p_no = torch.exp(log_no - log_binary)

    return (
        float(p_yes.detach().cpu()),
        float(p_no.detach().cpu()),
        float(p_yes_raw.detach().cpu()),
        float(p_no_raw.detach().cpu()),
    )


def compute_pyes_for_prompt(
    model,
    processor,
    model_config: ModelConfig,
    image: Any,
    prompt_text: str,
) -> Dict[str, Any]:
    """Run one forward pass and return P(yes/no) confidence fields."""
    device = _model_device(model)
    inputs = _format_vlm_prompt(processor, model_config, image, prompt_text)
    inputs = _move_to_device(inputs, device)

    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[:, -1, :]

    p_yes, p_no, p_yes_raw, p_no_raw = pyes_from_logits(
        next_token_logits,
        processor.tokenizer,
    )
    predicted = "yes" if p_yes >= p_no else "no"
    confidence = max(p_yes, p_no)
    yes_ids, no_ids = yes_no_token_ids(processor.tokenizer)

    return {
        "predicted": predicted,
        "confidence": confidence,
        "uncertainty": 1.0 - confidence,
        "p_yes": p_yes,
        "p_no": p_no,
        "p_yes_raw": p_yes_raw,
        "p_no_raw": p_no_raw,
        "yes_token_ids": yes_ids,
        "no_token_ids": no_ids,
    }


def compute_pyes_confidence(
    model,
    processor,
    model_config: ModelConfig,
    examples: Iterable[Mapping[str, Any]],
    *,
    question_type: str = "closed",
    prompt_template: str = PYES_PROMPT,
    show_progress: bool = True,
) -> List[PYesResult]:
    """Compute P(yes/no) confidence for closed-ended VQA examples."""
    if question_type != "closed":
        raise ValueError("P(yes/no) is only defined for closed yes/no questions")

    examples_list = list(examples)
    iterator = enumerate(examples_list)
    if show_progress:
        iterator = tqdm(iterator, total=len(examples_list), desc="P(yes/no)")

    results: List[PYesResult] = []
    for _, sample in iterator:
        question = sample["question"]
        prompt = prompt_template.format(question=question)
        scores = compute_pyes_for_prompt(
            model,
            processor,
            model_config,
            sample["image"],
            prompt,
        )

        answer = sample.get("answer")
        ground_truth = answer.lower().strip() if answer is not None else None
        predicted = scores["predicted"]
        is_correct = (
            check_answer_correct(predicted, ground_truth, "closed")
            if ground_truth is not None
            else None
        )

        results.append(PYesResult(
            question=question,
            ground_truth=ground_truth,
            predicted=predicted,
            confidence=float(scores["confidence"]),
            uncertainty=float(scores["uncertainty"]),
            p_yes=float(scores["p_yes"]),
            p_no=float(scores["p_no"]),
            p_yes_raw=float(scores["p_yes_raw"]),
            p_no_raw=float(scores["p_no_raw"]),
            is_correct=is_correct,
            answer_type=str(sample.get("answer_type", "closed")).lower(),
            prompt=prompt,
            yes_token_ids=list(scores["yes_token_ids"]),
            no_token_ids=list(scores["no_token_ids"]),
        ))

    return results


__all__ = [
    "PYES_PROMPT",
    "PYesResult",
    "yes_no_token_ids",
    "pyes_from_logits",
    "compute_pyes_for_prompt",
    "compute_pyes_confidence",
]
