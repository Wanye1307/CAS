"""
High-level API for Medical VQA calibration.

Provides four top-level functions:
  - load_dataset: Load a medical VQA dataset
  - load_model: Load a Vision-Language Model
  - compute_confidence: Compute sampling or verbalized confidence
  - evaluate_calibration: Compute ECE, MCE, and other calibration metrics
"""

import os

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
from datasets import Dataset

from .configs import (
    ModelConfig, DataConfig, InferenceConfig,
    ModelFamily, QuestionType, DatasetName,
)
from .data import get_dataset as _get_dataset, train_val_test_split
from .evaluation.calibration import compute_calibration_metrics, CalibrationPipeline


# =============================================================================
# Model Registry (short keys -> HuggingFace model IDs)
# =============================================================================

MODEL_REGISTRY: Dict[str, str] = {
    "qwen3vl_2b": "Qwen/Qwen3-VL-2B-Instruct",
    "qwen3vl_4b_local": os.path.join(os.environ.get("CAS_MODEL_ROOT", "models"), "Qwen3-VL-4B-Instruct"),
    "qwen3vl_8b": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen3vl_32b": "Qwen/Qwen3-VL-32B-Instruct",
    "internvl3_2b": "OpenGVLab/InternVL3-2B-hf",
    "internvl3_2b_local": os.path.join(os.environ.get("CAS_MODEL_ROOT", "models"), "InternVL3-2B-hf"),
    "internvl3_8b": "OpenGVLab/InternVL3-8B-hf",
    "internvl3_8b_local": os.path.join(os.environ.get("CAS_MODEL_ROOT", "models"), "InternVL3-8B-hf"),
    "internvl3_38b": "OpenGVLab/InternVL3-38B-hf",
    "llava_next_7b": "llava-hf/llava-v1.6-mistral-7b-hf",
    "llava_next_7b_local": os.path.join(os.environ.get("CAS_MODEL_ROOT", "models"), "llava-v1.6-mistral-7b-hf"),
    "llava_next_34b": "llava-hf/llava-v1.6-34b-hf",
    "llava_med": os.path.join(os.environ.get("CAS_MODEL_ROOT", "models"), "llava-med-v1.5-mistral-7b-hf"),
    "medgemma_4b": os.path.join(os.environ.get("CAS_MODEL_ROOT", "models"), "MedGemma-4B"),
    "lingshu_7b": os.path.join(os.environ.get("CAS_MODEL_ROOT", "models"), "Lingshu-7B"),
    "medvlm_r1": os.path.join(os.environ.get("CAS_MODEL_ROOT", "models"), "MedVLM-R1"),
}


# =============================================================================
# 1. load_dataset
# =============================================================================

def load_dataset(
    name: str,
    split: str = "test",
    question_type: str = "all",
    data_path: Optional[str] = None,
    subsample_size: Optional[int] = None,
    seed: int = 42,
) -> Dataset:
    """Load a medical VQA dataset.

    Supported datasets:
      - "vqa_rad" / "rad_vqa": Auto-downloads from HuggingFace (flaviagiammarino/vqa-rad)
      - "slake": Auto-downloads from HuggingFace (BoKelvin/SLAKE), English only
      - "vqa_med_2019": Requires manual download (set data_path)
      - "vqa_med_2020": Requires manual download (set data_path)
      - "vqa_med_2021": Requires manual download (set data_path)

    Args:
        name: Dataset name (e.g., "vqa_rad", "slake", "vqa_med_2019").
        split: Data split ("train", "test", "validation").
        question_type: Filter by question type ("all", "closed", "open").
        data_path: Path to local dataset files (required for VQA-Med datasets).
        subsample_size: Optional maximum number of samples.
        seed: Random seed for shuffling/subsampling.

    Returns:
        HuggingFace Dataset with columns: image, question, answer,
        answer_type, question_id, image_id, dataset_source.

    Example::

        import medvlm

        # Auto-download datasets
        dataset = medvlm.load_dataset("vqa_rad", split="test", question_type="closed")
        dataset = medvlm.load_dataset("slake", split="test", question_type="open")

        # Manual-download datasets
        dataset = medvlm.load_dataset(
            "vqa_med_2019", data_path="./data/vqa_med_2019/VQAMed2019Test"
        )
    """
    # Normalize name
    name_lower = name.lower().replace("-", "_")
    if name_lower == "rad_vqa":
        name_lower = "vqa_rad"

    try:
        dataset_name = DatasetName(name_lower)
    except ValueError:
        available = [d.value for d in DatasetName]
        raise ValueError(f"Unknown dataset: {name}. Available: {available}")

    try:
        qt = QuestionType(question_type)
    except ValueError:
        available = [q.value for q in QuestionType]
        raise ValueError(f"Unknown question_type: {question_type}. Available: {available}")

    config = DataConfig(
        dataset_name=dataset_name,
        question_type=qt,
        split=split,
        subsample_size=subsample_size,
        seed=seed,
        data_path=data_path,
    )

    dataset_wrapper = _get_dataset(config)
    return dataset_wrapper.load()


# =============================================================================
# 2. load_model
# =============================================================================

def load_model(
    model_name: str,
    quantization: Optional[str] = None,
    adapter_path: Optional[str] = None,
    use_flash_attention: bool = True,
    device_map: str = "auto",
) -> Tuple[Any, Any, ModelConfig]:
    """Load a Vision-Language Model.

    Args:
        model_name: Short key (e.g., "qwen3vl_8b") or full HuggingFace ID
                    (e.g., "Qwen/Qwen3-VL-8B-Instruct").
        quantization: None for full precision (default), "8bit" for 8-bit.
        adapter_path: Optional path to a LoRA adapter checkpoint.
        use_flash_attention: Whether to use Flash Attention 2 (default True).
        device_map: Device mapping strategy (default "auto").

    Returns:
        Tuple of (model, processor, model_config).

    Example::

        import medvlm

        model, processor, config = medvlm.load_model("qwen3vl_2b")
        model, processor, config = medvlm.load_model("qwen3vl_8b", quantization="8bit")
    """
    # Resolve model ID
    if model_name in MODEL_REGISTRY:
        model_id = MODEL_REGISTRY[model_name]
    else:
        model_id = model_name

    # Configure quantization
    use_4bit = False
    use_8bit = False
    if quantization == "8bit":
        use_8bit = True
    elif quantization == "4bit":
        use_4bit = True
    elif quantization is not None:
        raise ValueError(f"Unknown quantization: {quantization}. Use None, '8bit', or '4bit'.")

    model_config = ModelConfig(
        model_id=model_id,
        use_4bit=use_4bit,
        use_8bit=use_8bit,
        use_flash_attention=use_flash_attention,
        device_map=device_map,
    )

    from .models import load_model as _load_model
    model, processor = _load_model(model_config, adapter_path=adapter_path)
    model.eval()

    return model, processor, model_config


# =============================================================================
# 3. compute_confidence
# =============================================================================

def compute_confidence(
    model,
    processor,
    model_config: ModelConfig,
    examples: Iterable[Mapping],
    method: str = "sampling",
    question_type: str = "closed",
    # Sampling-specific
    num_samples: int = 20,
    temperature: float = 0.7,
    prompt_mode: str = "base",
    samples_per_batch: int = 25,
    # Verbalized-specific
    variant: str = "vanilla",
    max_new_tokens: int = 256,
    batch_size: int = 1,
    show_progress: bool = True,
) -> List:
    """Compute confidence scores for a set of examples.

    Args:
        model: Loaded VLM model.
        processor: Model processor/tokenizer.
        model_config: Model configuration (returned by load_model).
        examples: Iterable of dicts with keys ``image``, ``question``, and
            optionally ``answer``. A HuggingFace ``Dataset`` from
            ``load_dataset`` satisfies this contract, and a plain list of
            dicts works too. If ``answer`` is omitted, ``ground_truth`` and
            ``is_correct`` are set to ``None``.
        method: "sampling" or "verbalized".
        question_type: "closed" or "open".

        Sampling-specific args:
            num_samples: Number of samples per question (default 20).
            temperature: Sampling temperature (default 0.7).
            prompt_mode: "base" or "cot" (default "base").
            samples_per_batch: Max samples per generation call.

        Verbalized-specific args:
            variant: Prompt variant (default "vanilla"). One of:
                     "vanilla", "vanilla_cot", "punish", "top_k",
                     "two_stage", "linguistic".
            max_new_tokens: Max tokens per response (default 256).
            batch_size: Batch size for generation (default 1).

    Returns:
        List of SamplingResult or VerbalizedResult objects, each with:
          .question, .ground_truth, .predicted, .confidence, .is_correct

    Example::

        import medvlm

        dataset = medvlm.load_dataset("vqa_rad", split="test", question_type="closed")
        model, processor, config = medvlm.load_model("qwen3vl_2b")

        # Sampling-based confidence
        results = medvlm.compute_confidence(
            model, processor, config, dataset,
            method="sampling", num_samples=20,
        )

        # Verbalized confidence
        results = medvlm.compute_confidence(
            model, processor, config, dataset,
            method="verbalized", variant="vanilla_cot",
        )
    """
    if method == "sampling":
        from .confidence.sampling import compute_sampling_confidence
        return compute_sampling_confidence(
            model=model,
            processor=processor,
            model_config=model_config,
            examples=examples,
            num_samples=num_samples,
            temperature=temperature,
            prompt_mode=prompt_mode,
            question_type=question_type,
            samples_per_batch=samples_per_batch,
            show_progress=show_progress,
        )
    elif method == "verbalized":
        from .confidence.verbalized import compute_verbalized_confidence
        return compute_verbalized_confidence(
            model=model,
            processor=processor,
            model_config=model_config,
            examples=examples,
            variant=variant,
            question_type=question_type,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            show_progress=show_progress,
        )
    else:
        raise ValueError(f"Unknown method: {method}. Available: 'sampling', 'verbalized'.")


# =============================================================================
# 4. evaluate_calibration
# =============================================================================

def evaluate_calibration(
    correctness: Union[np.ndarray, List[bool]],
    confidences: Union[np.ndarray, List[float]],
    num_bins: int = 15,
    calibration_method: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate calibration metrics.

    Computes ECE, MCE, overconfidence, accuracy, and per-bin statistics.
    Optionally applies a post-hoc calibration method first.

    Args:
        correctness: Array of binary correctness labels.
        confidences: Array of confidence scores in [0, 1].
        num_bins: Number of bins for ECE calculation (default 15).
        calibration_method: Optional post-hoc calibration to apply before
                           computing metrics. One of: "temperature_scaling",
                           "platt", "isotonic", "histogram_binning".
                           If None (default), computes raw calibration metrics.

    Returns:
        Dictionary with keys: ece, mce, overconfidence, accuracy,
        mean_confidence, num_samples, bin_data.
        If calibration_method is set, also includes: calibrated_confidences,
        raw_metrics (metrics before calibration).

    Example::

        import medvlm
        import numpy as np

        # Raw calibration metrics
        report = medvlm.evaluate_calibration(
            correctness=[True, False, True, ...],
            confidences=[0.9, 0.8, 0.7, ...],
        )
        print(f"ECE: {report['ece']:.4f}, MCE: {report['mce']:.4f}")

        # With post-hoc calibration (leave-one-out on same data)
        report = medvlm.evaluate_calibration(
            correctness, confidences,
            calibration_method="temperature_scaling",
        )
    """
    correctness = np.asarray(correctness, dtype=np.float64)
    confidences = np.asarray(confidences, dtype=np.float64)

    if len(correctness) != len(confidences):
        raise ValueError("correctness and confidences must have the same length")

    # Build lightweight CalibrationResult-like objects for compute_calibration_metrics
    from .evaluation.calibration import CalibrationResult
    results = [
        CalibrationResult(
            question="", ground_truth="", predicted="",
            confidence=float(c), p_yes=float(c), p_no=1.0 - float(c),
            is_correct=bool(a), yes_count=0, no_count=0, unknown_count=0,
        )
        for c, a in zip(confidences, correctness)
    ]

    if calibration_method is None:
        return compute_calibration_metrics(results, num_bins)

    # Apply calibration
    pipeline = CalibrationPipeline(method=calibration_method, num_bins=num_bins)
    pipeline.fit(confidences, correctness)
    calibrated = pipeline.transform(confidences)

    # Compute raw metrics
    raw_metrics = compute_calibration_metrics(results, num_bins)

    # Compute calibrated metrics
    cal_results = [
        CalibrationResult(
            question="", ground_truth="", predicted="",
            confidence=float(c), p_yes=float(c), p_no=1.0 - float(c),
            is_correct=bool(a), yes_count=0, no_count=0, unknown_count=0,
        )
        for c, a in zip(calibrated, correctness)
    ]
    cal_metrics = compute_calibration_metrics(cal_results, num_bins)
    cal_metrics["calibrated_confidences"] = calibrated.tolist()
    cal_metrics["raw_metrics"] = raw_metrics

    return cal_metrics
