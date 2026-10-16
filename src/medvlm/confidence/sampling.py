"""
Sampling-based confidence estimation.

Generates N samples per question and computes empirical confidence
as P(majority answer) from the sample distribution.
"""

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional
from dataclasses import dataclass

import torch
from tqdm import tqdm

from ..configs import ModelConfig, ModelFamily


# =============================================================================
# Answer Parsing
# =============================================================================

def parse_answer_text(text: str) -> Optional[str]:
    """Extract answer from 'Answer: XXX' format.

    Returns normalized (lowercased, stripped) answer or None.
    """
    match = re.search(r"answer:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if match:
        answer = match.group(1).strip().rstrip(".")
        return answer.lower()

    # Fallback: short response (1-3 words), use the whole thing
    text_stripped = text.strip().rstrip(".")
    if len(text_stripped.split()) <= 3:
        return text_stripped.lower()

    return None


def normalize_answer(answer: Optional[str], question_type: str = "closed") -> Optional[str]:
    """Normalize parsed answer for comparison.

    For closed: map to yes/no.
    For open: lowercase strip.
    """
    if answer is None:
        return None

    answer = answer.lower().strip()

    if question_type == "closed":
        if answer in ("yes", "yes.", "yes,"):
            return "yes"
        if answer in ("no", "no.", "no,"):
            return "no"
        if answer.startswith("yes"):
            return "yes"
        if answer.startswith("no"):
            return "no"
        if "yes" in answer and "no" not in answer:
            return "yes"
        if "no" in answer and "yes" not in answer:
            return "no"
        return None

    return answer


def check_answer_correct(predicted: Optional[str], ground_truth: str,
                         question_type: str = "closed") -> bool:
    """Check if predicted answer matches ground truth.

    For closed: exact match (yes/no).
    For open: containment-based matching.
    """
    if predicted is None or predicted == "unknown":
        return False

    if question_type == "closed":
        return predicted == ground_truth

    pred = predicted.lower().strip()
    gt = ground_truth.lower().strip()

    if pred == gt:
        return True
    if gt in pred:
        return True
    if pred in gt:
        return True
    return False


# =============================================================================
# Prompt Formatting & Generation
# =============================================================================

BASE_SAMPLING_PROMPT = (
    "You are a medical AI assistant. Look at the provided medical image "
    "and answer the following question.\n\n"
    "Question: {question}\n\n"
    "Provide only the answer, without any explanation.\n"
    "Format:\n"
    "Answer: [your answer]"
)

COT_SAMPLING_PROMPT = (
    "You are a medical AI assistant. Look at the provided medical image "
    "and answer the following question.\n\n"
    "Question: {question}\n\n"
    "Think step by step. Then provide your answer.\n"
    "Format:\n"
    "Reasoning: [your reasoning]\n"
    "Answer: [your answer]"
)

SAMPLING_PROMPTS = {
    "base": BASE_SAMPLING_PROMPT,
    "cot": COT_SAMPLING_PROMPT,
}

PROMPT_MAX_TOKENS = {
    "base": 64,
    "cot": 256,
}



def _format_qwen_vl_prompt(processor, messages) -> Dict:
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
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    return processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

def _format_prompt(processor, model_config: ModelConfig, image: Any, prompt_text: str) -> Dict:
    """Format a single prompt for generation."""
    if model_config.model_family == ModelFamily.QWEN_VL:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ]}]
        inputs = _format_qwen_vl_prompt(processor, messages)
    elif model_config.model_family == ModelFamily.GEMMA:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        )
    elif model_config.model_family == ModelFamily.INTERNVL:
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt_text},
        ]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(images=image, text=text, return_tensors="pt")
    elif model_config.model_family in [ModelFamily.LLAVA, ModelFamily.LLAVA_NEXT]:
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt_text},
        ]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=text, images=image, return_tensors="pt")
    else:
        raise ValueError(f"Unsupported model family: {model_config.model_family}")
    return inputs


def _generate_samples(model, processor, inputs: Dict, num_sequences: int,
                      temperature: float = 0.7, max_new_tokens: int = 64) -> List[str]:
    """Generate multiple responses from a single prompt."""
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_sequences,
            pad_token_id=pad_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    responses = []
    for i in range(num_sequences):
        generated = outputs[i, prompt_len:]
        text = processor.tokenizer.decode(generated, skip_special_tokens=True)
        responses.append(text)
    return responses


# =============================================================================
# Main API
# =============================================================================

@dataclass
class SamplingResult:
    """Result of sampling-based confidence estimation for a single question."""
    question: str
    ground_truth: Optional[str]
    predicted: str
    confidence: float
    is_correct: Optional[bool]
    answer_counts: Dict[str, int]
    unknown_count: int
    valid_response_rate: float
    raw_responses: List[str]


def compute_sampling_confidence(
    model,
    processor,
    model_config: ModelConfig,
    examples: Iterable[Mapping],
    num_samples: int = 20,
    temperature: float = 0.7,
    prompt_mode: str = "base",
    question_type: str = "closed",
    samples_per_batch: int = 25,
    show_progress: bool = True,
) -> List[SamplingResult]:
    """Compute sampling-based confidence for a set of examples.

    Generates N samples per question, counts answer frequencies,
    and computes confidence as P(majority answer).

    Args:
        model: Loaded VLM model.
        processor: Model processor/tokenizer.
        model_config: Model configuration (for prompt formatting).
        examples: Iterable of dicts with keys ``image``, ``question``, and
            optionally ``answer`` (a HuggingFace ``Dataset`` satisfies this).
            If ``answer`` is omitted, ``ground_truth`` and ``is_correct`` are
            set to ``None``.
        num_samples: Number of samples per question (default 20).
        temperature: Sampling temperature (default 0.7).
        prompt_mode: "base" for direct answer, "cot" for chain-of-thought.
        question_type: "closed" or "open".
        samples_per_batch: Max samples per generation call (reduced on OOM).
        show_progress: Whether to show progress bar.

    Returns:
        List of SamplingResult objects.
    """
    prompt_template = SAMPLING_PROMPTS.get(prompt_mode)
    if prompt_template is None:
        raise ValueError(f"Unknown prompt_mode: {prompt_mode}. "
                        f"Available: {list(SAMPLING_PROMPTS.keys())}")
    max_new_tokens = PROMPT_MAX_TOKENS[prompt_mode]

    results = []
    total = len(examples) if hasattr(examples, "__len__") else None
    iterator = tqdm(examples, total=total, desc="Sampling confidence") if show_progress else examples
    eff_batch_size = samples_per_batch

    for sample in iterator:
        question = sample["question"]
        answer = sample.get("answer")
        ground_truth = answer.lower().strip() if answer is not None else None

        prompt_text = prompt_template.format(question=question)
        inputs = _format_prompt(processor, model_config, sample["image"], prompt_text)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        answer_counts = {}
        unknown_count = 0
        raw_responses = []

        samples_remaining = num_samples
        while samples_remaining > 0:
            batch_size = min(samples_per_batch, samples_remaining)
            responses = _generate_samples(
                model, processor, inputs, batch_size,
                temperature=temperature, max_new_tokens=max_new_tokens,
            )
            for response in responses:
                raw_responses.append(response)
                raw_answer = parse_answer_text(response)
                normalized = normalize_answer(raw_answer, question_type)
                if normalized is not None:
                    answer_counts[normalized] = answer_counts.get(normalized, 0) + 1
                else:
                    unknown_count += 1
            samples_remaining -= batch_size

        total = sum(answer_counts.values()) + unknown_count
        valid_count = sum(answer_counts.values())
        valid_response_rate = valid_count / total if total > 0 else 0

        if valid_count > 0:
            predicted = max(answer_counts, key=answer_counts.get)
            confidence = answer_counts[predicted] / valid_count
        else:
            predicted = "unknown"
            confidence = 0.5

        is_correct = (check_answer_correct(predicted, ground_truth, question_type)
                      if ground_truth is not None else None)

        results.append(SamplingResult(
            question=question,
            ground_truth=ground_truth,
            predicted=predicted,
            confidence=confidence,
            is_correct=is_correct,
            answer_counts=answer_counts,
            unknown_count=unknown_count,
            valid_response_rate=valid_response_rate,
            raw_responses=raw_responses,
        ))

    return results
