"""Whole-image ablation grounding scores for Medical VQA.

Core idea:
If a proposed answer is almost as likely without the real image as with the
image, the answer is weakly grounded and more likely to be a language-prior
hallucination.

Baselines:
- ``black``: replace the image with a same-size black image.
- ``noise``: replace the image with same-size Gaussian noise.
- ``textonly``: remove the image token entirely and ask the text question only.

Conventions:
- ``delta_log_target = log P(target | image) - log P(target | baseline)``.
  Larger values indicate stronger visual grounding.
- ``hallucination_score`` is a risk score where larger values are more
  suspicious. For closed questions it defaults to ``1 - |P_yes_img-P_yes_base|``.
  For open questions it defaults to ``-delta_log_target``.

Failure mode to monitor:
For existence questions whose correct answer is "no", the answer may genuinely
depend less on visible positive evidence. Always inspect the no-answer subset
separately and combine this score with confidence rather than using it as a
standalone decision rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import math

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from medvlm.configs import ModelConfig, ModelFamily
from medvlm.confidence.sampling import (
    BASE_SAMPLING_PROMPT,
    check_answer_correct,
    normalize_answer,
    parse_answer_text,
)


CLOSED_GROUNDING_PROMPT = (
    "You are a medical AI assistant. Look at the provided medical image "
    "and answer the following question with yes or no.\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

OPEN_GROUNDING_PROMPT = (
    "You are a medical AI assistant. Look at the provided medical image "
    "and answer the following question.\n\n"
    "Question: {question}\n\n"
    "Provide only the answer, without any explanation.\n\n"
    "Answer:"
)


@dataclass
class GroundingResult:
    """Whole-image ablation grounding result for one sample."""

    question: str
    ground_truth: Optional[str]
    target_answer: Optional[str]
    branch: str
    baseline: str
    answer_type: str
    is_correct: Optional[bool]

    # Core score. Higher is more visually grounded.
    delta_log_target: Optional[float]
    grounding_score: Optional[float]

    # Risk score. Higher is more suspicious / less grounded.
    hallucination_score: Optional[float]

    # Closed yes/no diagnostics.
    p_yes_image: Optional[float] = None
    p_yes_baseline: Optional[float] = None
    p_no_image: Optional[float] = None
    p_no_baseline: Optional[float] = None
    log_p_yes_image: Optional[float] = None
    log_p_yes_baseline: Optional[float] = None
    abs_p_yes_delta: Optional[float] = None
    delta_log_p_yes: Optional[float] = None
    target_prob_image: Optional[float] = None
    target_prob_baseline: Optional[float] = None

    # Open teacher-forced target diagnostics.
    mean_logprob_image: Optional[float] = None
    mean_logprob_baseline: Optional[float] = None
    sum_logprob_image: Optional[float] = None
    sum_logprob_baseline: Optional[float] = None
    target_token_count: Optional[int] = None

    is_negative_answer: bool = False
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


def _to_rgb_image(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    if hasattr(image, "convert"):
        return image.convert("RGB")
    raise TypeError(f"Unsupported image type for ablation: {type(image)!r}")


def make_baseline_image(
    image: Any,
    baseline: str = "black",
    *,
    noise_mean: float = 127.5,
    noise_std: float = 50.0,
    seed: Optional[int] = None,
) -> Optional[Image.Image]:
    """Create an image ablation baseline or return None for text-only."""
    baseline = baseline.lower().strip()
    if baseline == "textonly":
        return None

    pil = _to_rgb_image(image)
    if baseline == "black":
        return Image.new("RGB", pil.size, color=(0, 0, 0))
    if baseline == "noise":
        rng = np.random.default_rng(seed)
        arr = rng.normal(noise_mean, noise_std, size=(pil.height, pil.width, 3))
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")
    raise ValueError("baseline must be one of: 'black', 'noise', 'textonly'")


def _format_prompt(
    processor,
    model_config: ModelConfig,
    image: Optional[Any],
    prompt_text: str,
) -> Dict[str, Any]:
    """Format an image-text or text-only prompt for supported VLM families."""
    has_image = image is not None

    if model_config.model_family == ModelFamily.QWEN_VL:
        content = []
        if has_image:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt_text})
        messages = [{"role": "user", "content": content}]
        return processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

    if model_config.model_family == ModelFamily.GEMMA:
        content = []
        if has_image:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt_text})
        messages = [{"role": "user", "content": content}]
        return processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

    if model_config.model_family == ModelFamily.INTERNVL:
        content = []
        if has_image:
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt_text})
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if has_image:
            return processor(images=image, text=text, return_tensors="pt")
        return processor(text=text, return_tensors="pt")

    if model_config.model_family in (ModelFamily.LLAVA, ModelFamily.LLAVA_NEXT):
        content = []
        if has_image:
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt_text})
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if has_image:
            return processor(text=text, images=image, return_tensors="pt")
        return processor(text=text, return_tensors="pt")

    raise ValueError(f"Unsupported model family: {model_config.model_family}")


def _first_token_ids(tokenizer, candidates: Sequence[str]) -> List[int]:
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
    """Return first-token ids for yes/no comparison."""
    yes_ids = _first_token_ids(tokenizer, ("yes", "Yes", "YES", " yes", "\nyes"))
    no_ids = _first_token_ids(tokenizer, ("no", "No", "NO", " no", "\nno"))
    if not yes_ids or not no_ids:
        raise ValueError("Could not resolve yes/no token ids for this tokenizer")
    return yes_ids, no_ids


def _binary_next_token_probs(
    model,
    processor,
    model_config: ModelConfig,
    image: Optional[Any],
    prompt_text: str,
) -> Dict[str, float]:
    """Read normalized yes/no next-token probabilities."""
    device = _model_device(model)
    inputs = _format_prompt(processor, model_config, image, prompt_text)
    inputs = _move_to_device(inputs, device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :][0].float()

    yes_ids, no_ids = yes_no_token_ids(processor.tokenizer)
    probs = torch.softmax(logits, dim=-1)
    p_yes_raw = probs[yes_ids].sum()
    p_no_raw = probs[no_ids].sum()
    denom = p_yes_raw + p_no_raw
    if float(denom) <= 0:
        p_yes = torch.tensor(0.5, device=probs.device)
        p_no = torch.tensor(0.5, device=probs.device)
    else:
        p_yes = p_yes_raw / denom
        p_no = p_no_raw / denom

    eps = 1e-12
    return {
        "p_yes": float(p_yes.detach().cpu()),
        "p_no": float(p_no.detach().cpu()),
        "p_yes_raw": float(p_yes_raw.detach().cpu()),
        "p_no_raw": float(p_no_raw.detach().cpu()),
        "log_p_yes": float(torch.log(p_yes.clamp_min(eps)).detach().cpu()),
        "log_p_no": float(torch.log(p_no.clamp_min(eps)).detach().cpu()),
    }


def _append_target_tokens(
    processor,
    inputs: Dict[str, Any],
    target_answer: str,
    device: torch.device,
) -> Tuple[Dict[str, Any], torch.Tensor, int]:
    target_ids = processor.tokenizer.encode(
        str(target_answer).strip(),
        add_special_tokens=False,
    )
    if not target_ids:
        raise ValueError("target_answer produced no tokens")

    target = torch.tensor([target_ids], dtype=torch.long, device=device)
    prompt_len = int(inputs["input_ids"].shape[1])

    full_inputs = dict(inputs)
    full_inputs["input_ids"] = torch.cat([inputs["input_ids"], target], dim=1)
    if "attention_mask" in inputs:
        extra_mask = torch.ones(
            (inputs["attention_mask"].shape[0], target.shape[1]),
            dtype=inputs["attention_mask"].dtype,
            device=device,
        )
        full_inputs["attention_mask"] = torch.cat(
            [inputs["attention_mask"], extra_mask],
            dim=1,
        )
    if "token_type_ids" in inputs:
        extra_types = torch.zeros(
            (inputs["token_type_ids"].shape[0], target.shape[1]),
            dtype=inputs["token_type_ids"].dtype,
            device=device,
        )
        full_inputs["token_type_ids"] = torch.cat(
            [inputs["token_type_ids"], extra_types],
            dim=1,
        )
    if "mm_token_type_ids" in inputs:
        extra_mm_types = torch.zeros(
            (inputs["mm_token_type_ids"].shape[0], target.shape[1]),
            dtype=inputs["mm_token_type_ids"].dtype,
            device=device,
        )
        full_inputs["mm_token_type_ids"] = torch.cat(
            [inputs["mm_token_type_ids"], extra_mm_types],
            dim=1,
        )
    return full_inputs, target[0], prompt_len


def _target_logprob(
    model,
    processor,
    model_config: ModelConfig,
    image: Optional[Any],
    prompt_text: str,
    target_answer: str,
) -> Dict[str, float]:
    """Teacher-force a target answer and return target log-probabilities."""
    device = _model_device(model)
    prompt_inputs = _format_prompt(processor, model_config, image, prompt_text)
    prompt_inputs = _move_to_device(prompt_inputs, device)
    full_inputs, target_ids, prompt_len = _append_target_tokens(
        processor,
        prompt_inputs,
        target_answer,
        device,
    )

    with torch.no_grad():
        outputs = model(**full_inputs)
        logits = outputs.logits[0].float()

    token_logprobs = []
    for i, token_id in enumerate(target_ids):
        pos = prompt_len + i - 1
        log_probs = torch.log_softmax(logits[pos], dim=-1)
        token_logprobs.append(log_probs[int(token_id)])

    values = torch.stack(token_logprobs)
    return {
        "sum_logprob": float(values.sum().detach().cpu()),
        "mean_logprob": float(values.mean().detach().cpu()),
        "token_count": int(values.numel()),
    }


def generate_proposed_answer(
    model,
    processor,
    model_config: ModelConfig,
    example: Mapping[str, Any],
    *,
    branch: str = "closed",
    prompt_template: str = BASE_SAMPLING_PROMPT,
    max_new_tokens: int = 64,
) -> Tuple[str, str]:
    """Greedily generate a proposed answer for self-contained scoring."""
    device = _model_device(model)
    prompt_text = prompt_template.format(question=example["question"])
    inputs = _format_prompt(processor, model_config, example["image"], prompt_text)
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
    normalized = normalize_answer(parsed, branch)
    if normalized is None:
        normalized = parsed.strip().lower() if parsed else raw.strip().lower()
    if not normalized:
        normalized = "unknown"
    return normalized, raw


def _normalize_target(target_answer: Optional[str], branch: str) -> Optional[str]:
    if target_answer is None:
        return None
    text = str(target_answer).strip().lower()
    normalized = normalize_answer(text, branch)
    return normalized if normalized is not None else text


def _safe_log(value: float) -> float:
    return math.log(max(float(value), 1e-12))


def _is_negative_answer_text(value: Optional[str]) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"no", "none", "normal", "negative", "unremarkable"}:
        return True
    negative_phrases = (
        "no abnormality",
        "no abnormalities",
        "no acute abnormality",
        "no evidence",
        "without evidence",
        "not seen",
        "absent",
    )
    return any(phrase in text for phrase in negative_phrases)


def score(
    model,
    processor,
    model_config: ModelConfig,
    example: Mapping[str, Any],
    *,
    baseline: str = "black",
    branch: str = "closed",
    target_answer: Optional[str] = None,
    generate_if_missing: bool = True,
    prompt_template: Optional[str] = None,
    max_answer_tokens: int = 64,
    noise_seed: Optional[int] = None,
) -> GroundingResult:
    """Score one example with whole-image ablation.

    Args:
        model, processor, model_config: Loaded VLM triplet.
        example: Dict with at least ``image`` and ``question``. ``answer`` and
            ``answer_type`` are optional but used when available.
        baseline: ``"black"``, ``"noise"``, or ``"textonly"``.
        branch: ``"closed"`` or ``"open"``.
        target_answer: Proposed/model answer to evaluate. Passing this keeps
            the cost to two forward passes. If omitted and
            ``generate_if_missing=True``, a greedy answer is generated first.
        generate_if_missing: Generate a proposed answer if ``target_answer`` is
            missing.
        prompt_template: Optional prompt used for scoring. Closed defaults to
            yes/no prompt; open defaults to direct answer prompt.
        max_answer_tokens: Generation cap when target answer is not supplied.
        noise_seed: Random seed for the Gaussian-noise baseline.

    Returns:
        ``GroundingResult`` with both grounding and hallucination-risk scores.
    """
    branch = branch.lower().strip()
    if branch not in {"closed", "open"}:
        raise ValueError("branch must be 'closed' or 'open'")

    baseline_image = make_baseline_image(
        example["image"],
        baseline=baseline,
        seed=noise_seed,
    )
    image_prompt = prompt_template
    if image_prompt is None:
        image_prompt = CLOSED_GROUNDING_PROMPT if branch == "closed" else OPEN_GROUNDING_PROMPT
    prompt_text = image_prompt.format(question=example["question"])

    raw_answer = None
    target = _normalize_target(target_answer, branch)
    if target is None and generate_if_missing:
        target, raw_answer = generate_proposed_answer(
            model,
            processor,
            model_config,
            example,
            branch=branch,
            max_new_tokens=max_answer_tokens,
        )

    gt = example.get("answer")
    gt_norm = str(gt).strip().lower() if gt is not None else None
    is_correct = (
        check_answer_correct(target, gt_norm, branch)
        if target is not None and gt_norm is not None
        else None
    )
    is_negative = _is_negative_answer_text(target) or _is_negative_answer_text(gt_norm)

    if branch == "closed":
        image_probs = _binary_next_token_probs(
            model, processor, model_config, example["image"], prompt_text
        )
        base_probs = _binary_next_token_probs(
            model, processor, model_config, baseline_image, prompt_text
        )

        abs_delta = abs(image_probs["p_yes"] - base_probs["p_yes"])
        delta_log_p_yes = image_probs["log_p_yes"] - base_probs["log_p_yes"]

        target_prob_image = None
        target_prob_baseline = None
        delta_log_target = None
        if target == "yes":
            target_prob_image = image_probs["p_yes"]
            target_prob_baseline = base_probs["p_yes"]
            delta_log_target = image_probs["log_p_yes"] - base_probs["log_p_yes"]
        elif target == "no":
            target_prob_image = image_probs["p_no"]
            target_prob_baseline = base_probs["p_no"]
            delta_log_target = image_probs["log_p_no"] - base_probs["log_p_no"]

        hallucination_score = 1.0 - min(abs_delta, 1.0)
        grounding_score = abs_delta

        return GroundingResult(
            question=example["question"],
            ground_truth=gt_norm,
            target_answer=target,
            branch=branch,
            baseline=baseline,
            answer_type=str(example.get("answer_type", branch)).lower(),
            is_correct=is_correct,
            delta_log_target=delta_log_target,
            grounding_score=grounding_score,
            hallucination_score=hallucination_score,
            p_yes_image=image_probs["p_yes"],
            p_yes_baseline=base_probs["p_yes"],
            p_no_image=image_probs["p_no"],
            p_no_baseline=base_probs["p_no"],
            log_p_yes_image=image_probs["log_p_yes"],
            log_p_yes_baseline=base_probs["log_p_yes"],
            abs_p_yes_delta=abs_delta,
            delta_log_p_yes=delta_log_p_yes,
            target_prob_image=target_prob_image,
            target_prob_baseline=target_prob_baseline,
            is_negative_answer=is_negative,
            raw_answer=raw_answer,
        )

    if target is None:
        raise ValueError("target_answer is required for open grounding if generation is disabled")

    image_lp = _target_logprob(
        model, processor, model_config, example["image"], prompt_text, target
    )
    base_lp = _target_logprob(
        model, processor, model_config, baseline_image, prompt_text, target
    )
    delta = image_lp["mean_logprob"] - base_lp["mean_logprob"]

    return GroundingResult(
        question=example["question"],
        ground_truth=gt_norm,
        target_answer=target,
        branch=branch,
        baseline=baseline,
        answer_type=str(example.get("answer_type", branch)).lower(),
        is_correct=is_correct,
        delta_log_target=delta,
        grounding_score=delta,
        hallucination_score=-delta,
        mean_logprob_image=image_lp["mean_logprob"],
        mean_logprob_baseline=base_lp["mean_logprob"],
        sum_logprob_image=image_lp["sum_logprob"],
        sum_logprob_baseline=base_lp["sum_logprob"],
        target_token_count=image_lp["token_count"],
        is_negative_answer=is_negative,
        raw_answer=raw_answer,
    )


def compute_grounding_scores(
    model,
    processor,
    model_config: ModelConfig,
    examples: Iterable[Mapping[str, Any]],
    *,
    baseline: str = "black",
    branch: str = "closed",
    target_answers: Optional[Sequence[str]] = None,
    show_progress: bool = True,
    **kwargs,
) -> List[GroundingResult]:
    """Compute grounding scores for a sequence of examples."""
    examples_list = list(examples)
    if target_answers is not None and len(target_answers) != len(examples_list):
        raise ValueError("target_answers must have the same length as examples")

    iterator = enumerate(examples_list)
    if show_progress:
        iterator = tqdm(iterator, total=len(examples_list), desc=f"grounding/{branch}/{baseline}")

    results: List[GroundingResult] = []
    for idx, example in iterator:
        target = target_answers[idx] if target_answers is not None else None
        results.append(score(
            model,
            processor,
            model_config,
            example,
            baseline=baseline,
            branch=branch,
            target_answer=target,
            **kwargs,
        ))
    return results


def compute_grounding_scores_by_type(
    model,
    processor,
    model_config: ModelConfig,
    examples: Iterable[Mapping[str, Any]],
    *,
    baseline: str = "black",
    branch: str = "all",
    target_answers: Optional[Sequence[str]] = None,
    show_progress: bool = True,
    **kwargs,
) -> List[GroundingResult]:
    """Compute grounding scores, splitting closed/open when ``branch='all'``."""
    examples_list = list(examples)
    if branch != "all":
        return compute_grounding_scores(
            model,
            processor,
            model_config,
            examples_list,
            baseline=baseline,
            branch=branch,
            target_answers=target_answers,
            show_progress=show_progress,
            **kwargs,
        )

    if target_answers is not None and len(target_answers) != len(examples_list):
        raise ValueError("target_answers must have the same length as examples")

    results: List[GroundingResult] = []
    for qtype in ("closed", "open"):
        selected = [
            (i, sample) for i, sample in enumerate(examples_list)
            if str(sample.get("answer_type", "")).lower() == qtype
        ]
        if not selected:
            continue
        idxs, subset = zip(*selected)
        subset_targets = (
            [target_answers[i] for i in idxs]
            if target_answers is not None
            else None
        )
        results.extend(compute_grounding_scores(
            model,
            processor,
            model_config,
            subset,
            baseline=baseline,
            branch=qtype,
            target_answers=subset_targets,
            show_progress=show_progress,
            **kwargs,
        ))
    return results


__all__ = [
    "CLOSED_GROUNDING_PROMPT",
    "OPEN_GROUNDING_PROMPT",
    "GroundingResult",
    "make_baseline_image",
    "yes_no_token_ids",
    "score",
    "compute_grounding_scores",
    "compute_grounding_scores_by_type",
]
