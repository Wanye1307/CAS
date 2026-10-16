"""HEDGE / RadFlag / VASE hallucination scoring.

Thin wrapper around ``hedge_bench`` that runs the full HEDGE pipeline on a
medical VQA dataset:

  1. Greedy answer on the original image
  2. N high-temperature answers on the original image (with logprobs)
  3. One answer per distorted image (N distortions, with logprobs)
  4. Cluster answers (yes/no for closed, sentence-embedding for open) and
     compute SE, RadFlag, VASE via ``hedge_bench.algorithms``.

Public API::

    from medvlm import compute_hedge_scores

    results = compute_hedge_scores(
        model, processor, model_config, dataset,
        question_type="closed", n_samples=10, alpha=1.0,
    )
    h = np.array([r.VASE for r in results])
"""

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from ..configs import ModelConfig, ModelFamily
from .sampling import BASE_SAMPLING_PROMPT


# =============================================================================
# hedge_bench loader (defers import + stubs optuna for ABI issues in some envs)
# =============================================================================

def _load_hedge_bench():
    # hedge_bench transitively imports optuna, which can hit sqlite3 GLIBCXX
    # ABI issues on some torch/CUDA environments. Stub it out if absent.
    if "optuna" not in sys.modules:
        try:
            import optuna  # noqa: F401
        except Exception:
            sys.modules["optuna"] = types.ModuleType("optuna")

    from hedge_bench.algorithms import (
        sentence_semantic_entropy, radflag, vase, cluster_terms_by_embedding,
    )
    from hedge_bench.utils import distort_image
    return {
        "sentence_semantic_entropy": sentence_semantic_entropy,
        "radflag": radflag,
        "vase": vase,
        "cluster_terms_by_embedding": cluster_terms_by_embedding,
        "distort_image": distort_image,
    }


# =============================================================================
# Answer parsing & clustering
# =============================================================================

def _strip_answer_prefix(text: str) -> str:
    text = text.strip()
    for prefix in ("Answer:", "answer:", "ANSWER:", "Image:", "image:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text.strip("[]()\"' ")


def _parse_yes_no(text: str) -> str:
    t = _strip_answer_prefix(text).lower().strip("., !?")
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"
    return "other"


def cluster_by_yesno(answers: Sequence[str]) -> List[int]:
    """Cluster answers by parsed yes/no label (for closed questions)."""
    labels: dict = {}
    next_id = 0
    ids = []
    for a in answers:
        parsed = _parse_yes_no(a)
        if parsed not in labels:
            labels[parsed] = next_id
            next_id += 1
        ids.append(labels[parsed])
    return ids


_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def cluster_by_embedding(answers: Sequence[str], threshold: float = 0.85) -> List[int]:
    """Cluster answers by sentence embedding similarity (for open questions).

    Wraps ``hedge_bench.algorithms.cluster_terms_by_embedding`` with a
    lazily-loaded SentenceTransformer (``all-MiniLM-L6-v2``).
    """
    hb = _load_hedge_bench()
    embed = _get_embed_model()
    stripped = [_strip_answer_prefix(a) for a in answers]

    def embed_fn(text):
        return torch.from_numpy(embed.encode(text, convert_to_numpy=True))

    return hb["cluster_terms_by_embedding"](stripped, embed_fn, threshold=threshold)


# =============================================================================
# Prompt formatting (model-family specific, batched and single-image)
# =============================================================================


def _format_qwen_vl_messages(processor, messages, *, padding=False):
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
            padding=padding,
        )

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if isinstance(text, str):
        text = [text]
    image_inputs, video_inputs = process_vision_info(messages)
    return processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=padding,
        return_tensors="pt",
    )

def _format_single(processor, family: ModelFamily, image, question: str,
                   prompt_template: str):
    text = prompt_template.format(question=question)
    if family == ModelFamily.QWEN_VL:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": text},
        ]}]
        return _format_qwen_vl_messages(processor, messages)
    if family == ModelFamily.GEMMA:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": text},
        ]}]
        return processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
    if family == ModelFamily.INTERNVL:
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": text},
        ]}]
        text_f = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        return processor(images=image, text=text_f, return_tensors="pt")
    if family in (ModelFamily.LLAVA, ModelFamily.LLAVA_NEXT):
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": text},
        ]}]
        text_f = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        return processor(text=text_f, images=image, return_tensors="pt")
    raise ValueError(f"Unsupported model family: {family}")


def _format_batch(processor, family: ModelFamily, images, question: str,
                  prompt_template: str):
    text = prompt_template.format(question=question)
    # Decoder-only generation requires left-padding; right-padding silently
    # corrupts outputs for all but the longest sequence in the batch.
    tokenizer = getattr(processor, "tokenizer", processor)
    orig_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        if family == ModelFamily.QWEN_VL:
            conversations = [[{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": text},
                ],
            }] for img in images]
            return _format_qwen_vl_messages(processor, conversations, padding=True)
        if family == ModelFamily.GEMMA:
            conversations = [[{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": text},
                ],
            }] for img in images]
            return processor.apply_chat_template(
                conversations, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt", padding=True,
            )
        if family == ModelFamily.INTERNVL:
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": text},
            ]}]
            text_f = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            return processor(
                images=images, text=[text_f] * len(images),
                return_tensors="pt", padding=True,
            )
        if family in (ModelFamily.LLAVA, ModelFamily.LLAVA_NEXT):
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": text},
            ]}]
            text_f = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            return processor(
                text=[text_f] * len(images), images=images,
                return_tensors="pt", padding=True,
            )
        raise ValueError(f"Unsupported model family: {family}")
    finally:
        tokenizer.padding_side = orig_side


# =============================================================================
# Generation with log-likelihoods
# =============================================================================

def _extract_logprobs(outputs, prompt_len: int, pad_id: int, eos_id: int):
    gen_ids = outputs.sequences[:, prompt_len:]
    num_steps = len(outputs.scores)
    mean_log_probs = []
    for i in range(gen_ids.shape[0]):
        lps = []
        for step in range(min(num_steps, gen_ids.shape[1])):
            tid = gen_ids[i, step].item()
            if tid == pad_id or tid == eos_id:
                break
            lp = F.log_softmax(outputs.scores[step][i], dim=-1)
            lps.append(lp[tid].item())
        mean_log_probs.append(float(np.mean(lps)) if lps else -10.0)
    return gen_ids, mean_log_probs


def _generate(model, processor, inputs, num_return_sequences: int = 1,
              temperature: float = 0.7, max_new_tokens: int = 64,
              do_sample: bool = True):
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    tokenizer = getattr(processor, "tokenizer", processor)
    pad_id, eos_id = tokenizer.pad_token_id, tokenizer.eos_token_id

    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            temperature=temperature if do_sample else 1.0,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_id,
            output_scores=True,
            return_dict_in_generate=True,
        )
    gen_ids, lps = _extract_logprobs(out, prompt_len, pad_id, eos_id)
    texts = processor.batch_decode(gen_ids, skip_special_tokens=True)
    return texts, lps


def _generate_distorted_batch(model, processor, images, question: str,
                              prompt_template: str, family: ModelFamily,
                              temperature: float, max_new_tokens: int,
                              batch_size: int):
    """One answer per image, batched. Falls back to sequential on failure."""
    tokenizer = getattr(processor, "tokenizer", processor)
    pad_id, eos_id = tokenizer.pad_token_id, tokenizer.eos_token_id

    all_texts, all_lps = [], []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]
        try:
            inputs = _format_batch(processor, family, batch, question, prompt_template)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            prompt_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    do_sample=True,
                    num_return_sequences=1,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=pad_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            gen_ids, lps = _extract_logprobs(out, prompt_len, pad_id, eos_id)
            all_texts.extend(processor.batch_decode(gen_ids, skip_special_tokens=True))
            all_lps.extend(lps)
        except Exception:
            for img in batch:
                inp = _format_single(processor, family, img, question, prompt_template)
                t, lp = _generate(model, processor, inp,
                                  num_return_sequences=1,
                                  temperature=temperature,
                                  max_new_tokens=max_new_tokens)
                all_texts.append(t[0])
                all_lps.append(lp[0])
    return all_texts, all_lps


def _distort(image, n: int, distort_image_fn):
    w, h = image.size
    arr = np.array(image)
    transform = distort_image_fn(h, w)
    return [Image.fromarray(transform(image=arr)["image"]) for _ in range(n)]


# =============================================================================
# HEDGE metrics
# =============================================================================

def _compute_metrics(hb, greedy: str, orig_ans: List[str], orig_lps: List[float],
                     dist_ans: List[str], dist_lps: List[float], alpha: float,
                     cluster_fn: Callable):
    n = len(orig_ans)
    all_answers = [greedy] + orig_ans + dist_ans
    cluster_ids = cluster_fn(all_answers)

    ent_clean, dist_clean = hb["sentence_semantic_entropy"](
        orig_lps, cluster_ids[1:1 + n]
    )
    _ent_noisy, dist_noisy = hb["sentence_semantic_entropy"](
        dist_lps, cluster_ids[1 + n:]
    )
    vase_score = hb["vase"](n, cluster_ids, dist_clean, dist_noisy, alpha)
    radflag_score = hb["radflag"](cluster_ids, n)
    return {
        "SE": float(ent_clean),
        "RadFlag": float(radflag_score),
        "VASE": float(vase_score),
    }


# =============================================================================
# Public API
# =============================================================================

@dataclass
class HedgeResult:
    """HEDGE / RadFlag / VASE output for a single question."""
    question: str
    ground_truth: Optional[str]
    predicted: str
    is_correct: Optional[bool]
    greedy_answer: str
    SE: float
    RadFlag: float
    VASE: float
    original_high_temp: List[str]
    original_logprobs: List[float]
    distorted_high_temp: List[str]
    distorted_logprobs: List[float]


def compute_hedge_scores(
    model,
    processor,
    model_config: ModelConfig,
    examples: Iterable[Mapping],
    question_type: str = "closed",
    n_samples: int = 10,
    temperature: float = 0.7,
    max_new_tokens: int = 64,
    alpha: float = 1.0,
    batch_size: int = 5,
    prompt_template: Optional[str] = None,
    show_progress: bool = True,
) -> List[HedgeResult]:
    """Compute SE / RadFlag / VASE hallucination scores for a set of examples.

    Uses ``hedge_bench`` for clustering and scoring. For each question:
    generates one greedy answer and ``n_samples`` high-temp answers on the
    original image, plus one answer per distorted image (``n_samples``
    distortions), then clusters and scores.

    Args:
        model: Loaded VLM (e.g. from ``medvlm.load_model``).
        processor: Matching processor / tokenizer.
        model_config: ``ModelConfig`` (needed for prompt formatting per family).
        examples: Iterable of dicts with keys ``image``, ``question``, and
            optionally ``answer`` (a HuggingFace ``Dataset`` satisfies this).
            If ``answer`` is omitted, ``ground_truth`` and ``is_correct`` are
            set to ``None``.
        question_type: ``"closed"`` -> yes/no clustering; ``"open"`` -> sentence
            embedding clustering.
        n_samples: Number of high-temp samples (and distortions) per question.
        temperature: Sampling temperature.
        max_new_tokens: Generation cap per sample.
        alpha: VASE contrastive weight.
        batch_size: Distorted-image batch size for generation.
        prompt_template: Prompt (``"{question}"`` placeholder). Defaults to
            ``medvlm.confidence.sampling.BASE_SAMPLING_PROMPT``.
        show_progress: Show a tqdm bar.

    Returns:
        List of ``HedgeResult``, one per question.
    """
    hb = _load_hedge_bench()
    if prompt_template is None:
        prompt_template = BASE_SAMPLING_PROMPT

    cluster_fn = cluster_by_yesno if question_type == "closed" else cluster_by_embedding

    total = len(examples) if hasattr(examples, "__len__") else None
    iterator = enumerate(examples)
    if show_progress:
        iterator = tqdm(iterator, total=total, desc="HEDGE")

    results: List[HedgeResult] = []
    for _, example in iterator:
        image = example["image"]
        question = example["question"]
        gt = example.get("answer")

        inputs = _format_single(
            processor, model_config.model_family, image, question, prompt_template
        )

        greedy, _ = _generate(
            model, processor, inputs,
            num_return_sequences=1, max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        orig_ans, orig_lps = _generate(
            model, processor, inputs,
            num_return_sequences=n_samples,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        distorted = _distort(image, n_samples, hb["distort_image"])
        dist_ans, dist_lps = _generate_distorted_batch(
            model, processor, distorted, question, prompt_template,
            model_config.model_family, temperature, max_new_tokens, batch_size,
        )

        metrics = _compute_metrics(
            hb, greedy[0], orig_ans, orig_lps, dist_ans, dist_lps,
            alpha=alpha, cluster_fn=cluster_fn,
        )

        if question_type == "closed":
            predicted = _parse_yes_no(greedy[0])
            is_correct = (predicted == gt.strip().lower()) if gt is not None else None
        else:
            predicted = _strip_answer_prefix(greedy[0]).strip()
            is_correct = (predicted.lower() == gt.strip().lower()) if gt is not None else None

        results.append(HedgeResult(
            question=question,
            ground_truth=gt,
            predicted=predicted,
            is_correct=is_correct,
            greedy_answer=greedy[0],
            SE=metrics["SE"],
            RadFlag=metrics["RadFlag"],
            VASE=metrics["VASE"],
            original_high_temp=orig_ans,
            original_logprobs=orig_lps,
            distorted_high_temp=dist_ans,
            distorted_logprobs=dist_lps,
        ))
    return results
