"""
Verbalized confidence estimation.

Generates responses where the model self-reports its confidence level.
Supports 6 prompt variants: vanilla, vanilla_cot, punish, top_k,
two_stage, and linguistic.
"""

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
from dataclasses import dataclass

import torch
from tqdm import tqdm

from ..configs import ModelConfig, ModelFamily


# =============================================================================
# Prompt Templates
# =============================================================================

VERBALIZED_PROMPTS = {
    "vanilla": (
        "You are a medical AI assistant. Look at the provided medical image "
        "and answer the following question.\n\n"
        "Question: {question}\n\n"
        "Provide your answer and your confidence level (0-100%) in your answer.\n"
        "Format:\n"
        "Answer: [your answer]\n"
        "Confidence: [0-100]%"
    ),
    "vanilla_cot": (
        "You are a medical AI assistant. Look at the provided medical image "
        "and answer the following question.\n\n"
        "Question: {question}\n\n"
        "Think step by step. Then provide your answer and your confidence level "
        "(0-100%) in your answer.\n"
        "Format:\n"
        "Reasoning: [your reasoning]\n"
        "Answer: [your answer]\n"
        "Confidence: [0-100]%"
    ),
    "punish": (
        "You are a medical AI assistant. Look at the provided medical image "
        "and answer the following question. You will be punished if the answer "
        "is wrong but you answer it with high confidence.\n\n"
        "Question: {question}\n\n"
        "Provide your answer and your confidence level (0-100%) in your answer.\n"
        "Format:\n"
        "Answer: [your answer]\n"
        "Confidence: [0-100]%"
    ),
    "top_k": (
        "You are a medical AI assistant. Look at the provided medical image "
        "and answer the following question.\n\n"
        "Question: {question}\n\n"
        "Provide your top 3 best guesses for the answer, along with the "
        "probability (0-100%) that each guess is correct. The probabilities "
        "should sum to 100%.\n"
        "Format:\n"
        "Guess 1: [answer] (Probability: [X]%)\n"
        "Guess 2: [answer] (Probability: [Y]%)\n"
        "Guess 3: [answer] (Probability: [Z]%)"
    ),
    "two_stage_s1": (
        "You are a medical AI assistant. Look at the provided medical image "
        "and answer the following question.\n\n"
        "Question: {question}\n\n"
        "Provide your answer."
    ),
    "two_stage_s2": (
        "Question: {question}\n"
        "Proposed answer: {answer}\n\n"
        "How likely is the above answer to be correct? Provide a probability "
        "between 0% and 100%.\n"
        "Format:\n"
        "Confidence: [0-100]%"
    ),
    "linguistic": (
        "You are a medical AI assistant. Look at the provided medical image "
        "and answer the following question.\n\n"
        "Question: {question}\n\n"
        'Provide your answer and describe how confident you are using one of '
        'these terms: "almost certain", "highly likely", "very good chance", '
        '"probable", "likely", "better than even", "about even", "unlikely", '
        '"improbable", "very good chance not", "highly unlikely", '
        '"almost certainly not".\n'
        "Format:\n"
        "Answer: [your answer]\n"
        "Confidence: [one of the terms above]"
    ),
}

VERBALIZED_VARIANTS = ["vanilla", "vanilla_cot", "punish", "top_k", "two_stage", "linguistic"]

LINGUISTIC_MAP = {
    "almost certain": 0.95,
    "highly likely": 0.90,
    "very good chance": 0.85,
    "probable": 0.75,
    "likely": 0.70,
    "better than even": 0.60,
    "about even": 0.50,
    "unlikely": 0.30,
    "improbable": 0.20,
    "very good chance not": 0.15,
    "highly unlikely": 0.10,
    "almost certainly not": 0.05,
}


# =============================================================================
# Answer & Confidence Parsing
# =============================================================================

def parse_answer_yes_no(text: str) -> Optional[str]:
    """Extract yes/no answer from verbalized response."""
    text_lower = text.lower().strip()

    match = re.search(r"answer:\s*(yes|no)\b", text_lower)
    if match:
        return match.group(1)

    if text_lower in ("yes", "no", "yes.", "no."):
        return text_lower.rstrip(".")

    if text_lower.startswith("yes"):
        return "yes"
    if text_lower.startswith("no"):
        return "no"

    has_yes = "yes" in text_lower
    has_no = "no" in text_lower
    if has_yes and not has_no:
        return "yes"
    if has_no and not has_yes:
        return "no"

    return None


def parse_answer_general(text: str) -> Optional[str]:
    """Extract answer text from response (any question type)."""
    match = re.search(r"answer:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if match:
        answer = match.group(1).strip().rstrip(".")
        return answer.lower()

    text_stripped = text.strip().rstrip(".")
    if len(text_stripped.split()) <= 5:
        return text_stripped.lower()

    return None


def parse_confidence_numeric(text: str) -> Optional[float]:
    """Extract numeric confidence from 'Confidence: XX%' pattern.

    Returns float in [0, 1] or None if not found.
    """
    match = re.search(r"confidence:\s*(\d+(?:\.\d+)?)\s*%?", text.lower())
    if match:
        val = float(match.group(1))
        if val > 1.0:
            val /= 100.0
        return max(0.0, min(1.0, val))
    return None


def parse_confidence_linguistic(text: str, mapping: Optional[Dict[str, float]] = None) -> Optional[float]:
    """Map linguistic confidence term to numerical value."""
    if mapping is None:
        mapping = LINGUISTIC_MAP

    text_lower = text.lower()
    conf_match = re.search(r"confidence:\s*(.+?)(?:\.|$)", text_lower)
    search_text = conf_match.group(1).strip() if conf_match else text_lower

    for term in sorted(mapping.keys(), key=len, reverse=True):
        if term in search_text:
            return mapping[term]

    return None


def _parse_top_k(text: str, force_yes_no: bool = True) -> Tuple[Optional[str], Optional[float]]:
    """Extract top guess and its probability from top-K format."""
    matches = re.findall(
        r"guess\s*\d+:\s*(.*?)\s*\(probability:\s*(\d+(?:\.\d+)?)\s*%?\)",
        text.lower()
    )
    if matches:
        best_answer, best_prob = None, -1
        for answer, prob_str in matches:
            prob = float(prob_str)
            if prob > 1.0:
                prob /= 100.0
            if prob > best_prob:
                best_prob = prob
                best_answer = answer.strip()
            prob = max(0.0, min(1.0, prob))
        if force_yes_no:
            ans = parse_answer_yes_no(best_answer) if best_answer else None
        else:
            ans = best_answer
        return ans, max(0.0, min(1.0, best_prob)) if best_prob >= 0 else None
    return None, None


def _check_answer_correct(predicted: Optional[str], ground_truth: str,
                          question_type: str = "closed") -> bool:
    """Check if predicted answer matches ground truth."""
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

def _format_prompts_batch(processor, model_config: ModelConfig,
                          images: List[Any], prompt_texts: List[str]) -> Dict:
    """Format and tokenize a batch of image+text prompts."""
    texts = []
    for image, prompt_text in zip(images, prompt_texts):
        if model_config.model_family == ModelFamily.QWEN_VL:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ]}]
        else:
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        texts.append(text)

    orig_padding_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"

    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt",
    )

    processor.tokenizer.padding_side = orig_padding_side
    return inputs


def _generate_batch(model, processor, inputs: Dict,
                    max_new_tokens: int = 256,
                    do_sample: bool = False,
                    temperature: float = 1.0) -> List[str]:
    """Generate responses for a batch of inputs."""
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

    with torch.no_grad():
        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": pad_token_id,
        }
        if do_sample:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        outputs = model.generate(**gen_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    responses = []
    for i in range(outputs.shape[0]):
        generated = outputs[i, prompt_len:]
        text = processor.tokenizer.decode(generated, skip_special_tokens=True)
        responses.append(text)
    return responses


def _parse_variant_response(variant: str, response: str,
                            question_type: str = "closed") -> Tuple[Optional[str], Optional[float]]:
    """Parse predicted answer and confidence from a response."""
    use_yes_no = (question_type == "closed")

    if variant == "top_k":
        predicted, confidence = _parse_top_k(response, force_yes_no=use_yes_no)
        if predicted is None:
            predicted = parse_answer_yes_no(response) if use_yes_no else parse_answer_general(response)
    elif variant == "linguistic":
        predicted = parse_answer_yes_no(response) if use_yes_no else parse_answer_general(response)
        confidence = parse_confidence_linguistic(response)
    else:
        predicted = parse_answer_yes_no(response) if use_yes_no else parse_answer_general(response)
        confidence = parse_confidence_numeric(response)
    return predicted, confidence


# =============================================================================
# Main API
# =============================================================================

@dataclass
class VerbalizedResult:
    """Result of verbalized confidence estimation for a single question."""
    question: str
    ground_truth: Optional[str]
    predicted: str
    confidence: float
    is_correct: Optional[bool]
    parse_success: bool
    raw_response: str
    variant: str
    raw_stage2: Optional[str] = None


def compute_verbalized_confidence(
    model,
    processor,
    model_config: ModelConfig,
    examples: Iterable[Mapping],
    variant: str = "vanilla",
    question_type: str = "closed",
    max_new_tokens: int = 256,
    batch_size: int = 1,
    show_progress: bool = True,
) -> List[VerbalizedResult]:
    """Compute verbalized confidence for a set of examples.

    The model is prompted to self-report its confidence in its answer.

    Args:
        model: Loaded VLM model.
        processor: Model processor/tokenizer.
        model_config: Model configuration (for prompt formatting).
        examples: Iterable of dicts with keys ``image``, ``question``, and
            optionally ``answer`` (a HuggingFace ``Dataset`` satisfies this).
            If ``answer`` is omitted, ``ground_truth`` and ``is_correct`` are
            set to ``None``.
        variant: Prompt variant. One of: "vanilla", "vanilla_cot", "punish",
                 "top_k", "two_stage", "linguistic".
        question_type: "closed" or "open".
        max_new_tokens: Maximum tokens to generate per response.
        batch_size: Batch size for generation.
        show_progress: Whether to show progress bar.

    Returns:
        List of VerbalizedResult objects.
    """
    if variant not in VERBALIZED_VARIANTS:
        raise ValueError(f"Unknown variant: {variant}. "
                        f"Available: {VERBALIZED_VARIANTS}")

    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    results = []
    examples_list = list(examples)
    n = len(examples_list)

    for batch_start in tqdm(range(0, n, batch_size),
                            desc=f"Verbalized ({variant})",
                            disable=not show_progress):
        batch_end = min(batch_start + batch_size, n)
        batch_samples = examples_list[batch_start:batch_end]

        questions = [s["question"] for s in batch_samples]
        ground_truths = [s["answer"].lower().strip() if s.get("answer") is not None else None
                         for s in batch_samples]
        images = [s["image"] for s in batch_samples]

        if variant == "two_stage":
            # Stage 1: generate answers
            s1_prompts = [VERBALIZED_PROMPTS["two_stage_s1"].format(question=q)
                          for q in questions]
            s1_inputs = _format_prompts_batch(processor, model_config, images, s1_prompts)
            s1_inputs = {k: v.to(model.device) for k, v in s1_inputs.items()}
            s1_responses = _generate_batch(model, processor, s1_inputs, max_new_tokens)

            if question_type == "closed":
                s1_answers = [parse_answer_yes_no(r) or r.strip() for r in s1_responses]
            else:
                s1_answers = [parse_answer_general(r) or r.strip() for r in s1_responses]

            # Stage 2: generate confidence
            s2_prompts = [VERBALIZED_PROMPTS["two_stage_s2"].format(question=q, answer=a)
                          for q, a in zip(questions, s1_answers)]
            s2_inputs = _format_prompts_batch(processor, model_config, images, s2_prompts)
            s2_inputs = {k: v.to(model.device) for k, v in s2_inputs.items()}
            s2_responses = _generate_batch(model, processor, s2_inputs, max_new_tokens=256)

            for i in range(len(batch_samples)):
                if question_type == "closed":
                    predicted = parse_answer_yes_no(s1_responses[i])
                else:
                    predicted = parse_answer_general(s1_responses[i])
                confidence = parse_confidence_numeric(s2_responses[i])
                if predicted is None:
                    predicted = "unknown"
                parse_success = confidence is not None
                if confidence is None:
                    confidence = 0.5

                results.append(VerbalizedResult(
                    question=questions[i],
                    ground_truth=ground_truths[i],
                    predicted=predicted,
                    confidence=confidence,
                    is_correct=(_check_answer_correct(predicted, ground_truths[i], question_type)
                                if ground_truths[i] is not None else None),
                    parse_success=parse_success,
                    raw_response=s1_responses[i],
                    variant=variant,
                    raw_stage2=s2_responses[i],
                ))
        else:
            prompts = [VERBALIZED_PROMPTS[variant].format(question=q)
                       for q in questions]
            inputs = _format_prompts_batch(processor, model_config, images, prompts)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            responses = _generate_batch(model, processor, inputs, max_new_tokens)

            for i in range(len(batch_samples)):
                predicted, confidence = _parse_variant_response(
                    variant, responses[i], question_type
                )
                if predicted is None:
                    predicted = "unknown"
                parse_success = confidence is not None
                if confidence is None:
                    confidence = 0.5

                results.append(VerbalizedResult(
                    question=questions[i],
                    ground_truth=ground_truths[i],
                    predicted=predicted,
                    confidence=confidence,
                    is_correct=(_check_answer_correct(predicted, ground_truths[i], question_type)
                                if ground_truths[i] is not None else None),
                    parse_success=parse_success,
                    raw_response=responses[i],
                    variant=variant,
                ))

    return results
