"""MedVLM — inference, confidence, and calibration in Medical VQA.

Quick start::

    import medvlm

    # Load data and model
    dataset = medvlm.load_dataset("vqa_rad", split="test", question_type="closed")
    model, processor, config = medvlm.load_model("qwen3vl_2b")

    # Compute confidence
    results = medvlm.compute_confidence(
        model, processor, config, dataset, method="sampling", num_samples=20,
    )

    # Evaluate calibration
    report = medvlm.evaluate_calibration(
        [r.is_correct for r in results], [r.confidence for r in results],
    )
    print(f"ECE: {report['ece']:.4f}")

    # Post-hoc calibration (HAC)
    from medvlm import CalibrationPipeline
    cal = CalibrationPipeline(method="hac_platt")
    cal.fit(val_conf, val_corr, hallucination_scores=val_h)
    calibrated = cal.transform(test_conf, hallucination_scores=test_h)
"""

__version__ = "0.1.0"

# -- High-level API --
from .api import (
    load_dataset,
    load_model,
    compute_confidence,
    evaluate_calibration,
    MODEL_REGISTRY,
)

# -- Hallucination scoring --
from .confidence import compute_hedge_scores, HedgeResult

# -- Calibration --
from .evaluation import CalibrationPipeline

# -- Data utilities --
from .data import train_val_test_split

__all__ = [
    # API
    "load_dataset",
    "load_model",
    "compute_confidence",
    "evaluate_calibration",
    "MODEL_REGISTRY",
    # Hallucination scoring
    "compute_hedge_scores",
    "HedgeResult",
    # Calibration
    "CalibrationPipeline",
    # Data
    "train_val_test_split",
]
