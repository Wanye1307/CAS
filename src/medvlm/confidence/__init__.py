"""Confidence estimation methods for Medical VQA.

Provides sampling-based and verbalized confidence estimation for VLMs.
"""

from .sampling import (
    compute_sampling_confidence,
    parse_answer_text,
    normalize_answer,
    check_answer_correct,
)
from .verbalized import (
    compute_verbalized_confidence,
    parse_confidence_numeric,
    parse_confidence_linguistic,
    VERBALIZED_PROMPTS,
    LINGUISTIC_MAP,
)
from .hedge import (
    compute_hedge_scores,
    HedgeResult,
    cluster_by_yesno,
    cluster_by_embedding,
)

__all__ = [
    "compute_sampling_confidence",
    "parse_answer_text",
    "normalize_answer",
    "check_answer_correct",
    "compute_verbalized_confidence",
    "parse_confidence_numeric",
    "parse_confidence_linguistic",
    "VERBALIZED_PROMPTS",
    "LINGUISTIC_MAP",
    "compute_hedge_scores",
    "HedgeResult",
    "cluster_by_yesno",
    "cluster_by_embedding",
]
