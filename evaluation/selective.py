"""Selective prediction metrics for Medical VQA.

Conventions are fixed intentionally:

- ``correctness`` uses 1/True for correct answers and 0/False for wrong answers.
- The error/hallucination label is derived as ``error = 1 - correctness``.
- AUROC is always computed as ``roc_auc_score(error, uncertainty_score)``:
  label 1 means wrong/hallucinated, and a larger score means more uncertain.
- If the input signal is a confidence score, it is converted to uncertainty as
  ``1 - confidence`` for AUROC. Selective answering still sorts by confidence
  from high to low.

This avoids the common ambiguity where ``roc_auc_score(correct, confidence)``
and ``roc_auc_score(error, uncertainty)`` are complementary under the same
ranking.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.metrics import roc_auc_score


DEFAULT_RISK_LEVELS: Tuple[float, ...] = (0.01, 0.05, 0.10)
ScoreInput = Union[Sequence[float], np.ndarray]
Report = Dict[str, Any]


def _as_float_array(values: ScoreInput, name: str) -> np.ndarray:
    """Convert an array-like input to a 1-D float array, preserving None as NaN."""
    raw = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=object)
    arr = np.array([np.nan if v is None else float(v) for v in raw.ravel()], dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return arr


def _as_object_array(values: Sequence[Any], name: str) -> np.ndarray:
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=object).ravel()
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return arr


def _as_binary_array(values: np.ndarray, name: str) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return values.astype(bool)
    unique = set(np.unique(finite).tolist())
    if not unique.issubset({0.0, 1.0}):
        raise ValueError(f"{name} must contain only binary values 0/1 or bool")
    return values.astype(bool)


def _prepare_arrays(
    correctness: ScoreInput,
    scores: ScoreInput,
    *,
    hallucination_labels: Optional[ScoreInput] = None,
    question_types: Optional[Sequence[Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Validate inputs, drop rows with missing numeric values, and align arrays."""
    correctness_arr = _as_float_array(correctness, "correctness")
    score_arr = _as_float_array(scores, "scores")
    if correctness_arr.size != score_arr.size:
        raise ValueError("correctness and scores must have the same length")

    valid = np.isfinite(correctness_arr) & np.isfinite(score_arr)

    if hallucination_labels is not None:
        hallucination_arr = _as_float_array(hallucination_labels, "hallucination_labels")
        if hallucination_arr.size != score_arr.size:
            raise ValueError("hallucination_labels must have the same length as scores")
        valid &= np.isfinite(hallucination_arr)
    else:
        hallucination_arr = None

    if question_types is not None:
        type_arr = _as_object_array(question_types, "question_types")
        if type_arr.size != score_arr.size:
            raise ValueError("question_types must have the same length as scores")
        valid &= np.array([v is not None for v in type_arr], dtype=bool)
    else:
        type_arr = None

    correctness_arr = correctness_arr[valid]
    score_arr = score_arr[valid]
    corr_bool = _as_binary_array(correctness_arr, "correctness")

    if hallucination_arr is not None:
        hall_bool = _as_binary_array(hallucination_arr[valid], "hallucination_labels")
    else:
        hall_bool = ~corr_bool

    if type_arr is not None:
        type_arr = np.array([str(v).strip().lower() for v in type_arr[valid]], dtype=object)

    return corr_bool, score_arr, hall_bool, type_arr


def _score_views(scores: np.ndarray, score_type: str) -> Tuple[np.ndarray, np.ndarray, str]:
    """Return uncertainty score, selection score, and threshold rule."""
    key = score_type.lower().strip()
    if key in {"confidence", "conf", "correctness", "prob_correct", "p_correct"}:
        uncertainty = 1.0 - scores
        selection_score = scores
        rule = "score >= threshold"
    elif key in {"uncertainty", "error", "hallucination", "risk", "u"}:
        uncertainty = scores
        selection_score = -scores
        rule = "score <= threshold"
    else:
        raise ValueError(
            "score_type must be 'confidence' or 'uncertainty' "
            f"(got {score_type!r})"
        )
    return uncertainty, selection_score, rule


def _safe_float(value: float) -> float:
    return float(value) if np.isfinite(value) else float("nan")


def auroc_error_detection(
    correctness: ScoreInput,
    scores: ScoreInput,
    *,
    score_type: str = "confidence",
) -> float:
    """Compute AUROC for detecting wrong/hallucinated answers.

    This function uses the fixed convention:
    ``roc_auc_score(error_label, uncertainty_score)``, where
    ``error_label=1`` means the answer is wrong/hallucinated and larger
    uncertainty scores should rank earlier as likely failures.

    Args:
        correctness: Binary correctness labels, 1/True means correct.
        scores: Confidence or uncertainty scores.
        score_type: ``"confidence"`` if larger scores mean more likely correct,
            or ``"uncertainty"`` if larger scores mean more likely wrong.

    Returns:
        AUROC in [0, 1], or NaN if only one class is present.
    """
    corr_bool, score_arr, _, _ = _prepare_arrays(correctness, scores)
    if corr_bool.size == 0:
        return float("nan")

    uncertainty, _, _ = _score_views(score_arr, score_type)
    error = (~corr_bool).astype(int)
    if np.unique(error).size < 2:
        return float("nan")
    return float(roc_auc_score(error, uncertainty))


def risk_coverage_curve(
    correctness: ScoreInput,
    scores: ScoreInput,
    *,
    score_type: str = "confidence",
    hallucination_labels: Optional[ScoreInput] = None,
) -> Dict[str, np.ndarray]:
    """Build selective risk, hallucination, and commission/omission curves.

    Samples are sorted from most likely answerable to least likely answerable:
    by descending confidence for confidence scores, or by ascending uncertainty
    for uncertainty scores.

    The kth point answers the first k sorted samples, so
    ``coverage = k / n`` and ``risk = errors_answered / k``.

    Commission/omission definitions:
    - commission: a wrong answer is accepted.
    - omission: a correct answer is rejected/abstained.

    Returns numpy arrays so callers can plot or integrate directly.
    """
    corr_bool, score_arr, hall_bool, _ = _prepare_arrays(
        correctness,
        scores,
        hallucination_labels=hallucination_labels,
    )
    n = int(corr_bool.size)
    if n == 0:
        raise ValueError("No valid samples after dropping missing values")

    _, selection_score, rule = _score_views(score_arr, score_type)
    order = np.argsort(-selection_score, kind="mergesort")

    sorted_correct = corr_bool[order].astype(int)
    sorted_error = 1 - sorted_correct
    sorted_hallucination = hall_bool[order].astype(int)
    sorted_scores = score_arr[order]
    sorted_selection = selection_score[order]

    answered = np.arange(1, n + 1, dtype=np.float64)
    coverage = answered / float(n)

    cum_correct = np.cumsum(sorted_correct)
    cum_error = np.cumsum(sorted_error)
    cum_hallucination = np.cumsum(sorted_hallucination)

    risk = cum_error / answered
    hallucination_rate = cum_hallucination / answered

    total_correct = int(sorted_correct.sum())
    omission_count = total_correct - cum_correct
    commission_count = cum_error
    abstained = n - answered

    omission_rate_abstained = np.divide(
        omission_count,
        abstained,
        out=np.zeros_like(answered),
        where=abstained > 0,
    )

    return {
        "coverage": coverage,
        "risk": risk,
        "hallucination_rate": hallucination_rate,
        "score_threshold": sorted_scores,
        "selection_score": sorted_selection,
        "answered_count": answered.astype(int),
        "commission_count": commission_count.astype(int),
        "omission_count": omission_count.astype(int),
        "commission_rate_total": commission_count / float(n),
        "omission_rate_total": omission_count / float(n),
        "commission_rate_answered": risk,
        "omission_rate_abstained": omission_rate_abstained,
        "sort_index": order.astype(int),
        "threshold_rule": np.array([rule] * n, dtype=object),
    }


def aurc_from_curve(curve: Mapping[str, np.ndarray], *, rate_key: str = "risk") -> float:
    """Compute AURC from a risk-coverage or hallucination-coverage curve.

    The discrete selective-classification convention is used:
    ``AURC = mean(risk_k for k=1..n)`` where each prefix adds area ``1/n``.
    Lower is better.
    """
    values = np.asarray(curve[rate_key], dtype=np.float64)
    if values.size == 0:
        return float("nan")
    return float(values.mean())


def ideal_aurc(correctness: ScoreInput) -> float:
    """AURC of an ideal sorter that answers all correct samples first."""
    corr_arr = _as_float_array(correctness, "correctness")
    valid = np.isfinite(corr_arr)
    corr_bool = _as_binary_array(corr_arr[valid], "correctness")
    if corr_bool.size == 0:
        return float("nan")

    ideal_errors = np.sort((~corr_bool).astype(int))
    answered = np.arange(1, ideal_errors.size + 1, dtype=np.float64)
    ideal_risk = np.cumsum(ideal_errors) / answered
    return float(ideal_risk.mean())


def aurc(
    correctness: ScoreInput,
    scores: ScoreInput,
    *,
    score_type: str = "confidence",
) -> Dict[str, float]:
    """Compute AURC, ideal AURC, and E-AURC for selective answering."""
    corr_bool, _, _, _ = _prepare_arrays(correctness, scores)
    curve = risk_coverage_curve(correctness, scores, score_type=score_type)
    observed = aurc_from_curve(curve, rate_key="risk")
    ideal = ideal_aurc(corr_bool.astype(int))
    excess = observed - ideal
    if np.isfinite(excess) and excess < 0 and abs(excess) < 1e-12:
        excess = 0.0
    return {
        "aurc": _safe_float(observed),
        "ideal_aurc": _safe_float(ideal),
        "eaurc": _safe_float(excess),
    }


def coverage_at_risk(
    curve: Mapping[str, np.ndarray],
    *,
    risk_levels: Sequence[float] = DEFAULT_RISK_LEVELS,
    rate_key: str = "risk",
) -> Dict[str, Dict[str, Optional[float]]]:
    """Return the largest prefix coverage whose risk is at most each level."""
    coverage = np.asarray(curve["coverage"], dtype=np.float64)
    rate = np.asarray(curve[rate_key], dtype=np.float64)
    thresholds = np.asarray(curve["score_threshold"], dtype=np.float64)
    rules = curve.get("threshold_rule")
    rule = str(rules[0]) if isinstance(rules, np.ndarray) and rules.size else None

    out: Dict[str, Dict[str, Optional[float]]] = {}
    for level in risk_levels:
        ok = np.where(rate <= float(level) + 1e-12)[0]
        key = f"{float(level):.4g}"
        if ok.size == 0:
            out[key] = {
                "coverage": 0.0,
                "risk": None,
                "score_threshold": None,
                "threshold_rule": rule,
            }
            continue
        idx = int(ok[-1])
        out[key] = {
            "coverage": float(coverage[idx]),
            "risk": float(rate[idx]),
            "score_threshold": float(thresholds[idx]),
            "threshold_rule": rule,
        }
    return out


def commission_omission_curve(
    correctness: ScoreInput,
    scores: ScoreInput,
    *,
    score_type: str = "confidence",
) -> Dict[str, np.ndarray]:
    """Return commission/omission arrays for each selective coverage prefix."""
    curve = risk_coverage_curve(correctness, scores, score_type=score_type)
    return {
        "coverage": curve["coverage"],
        "score_threshold": curve["score_threshold"],
        "commission_count": curve["commission_count"],
        "omission_count": curve["omission_count"],
        "commission_rate_total": curve["commission_rate_total"],
        "omission_rate_total": curve["omission_rate_total"],
        "commission_rate_answered": curve["commission_rate_answered"],
        "omission_rate_abstained": curve["omission_rate_abstained"],
    }


def _serialise_curve(curve: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in curve.items():
        if key == "threshold_rule":
            continue
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        else:
            out[key] = value
    return out


def evaluate_selective(
    correctness: ScoreInput,
    scores: ScoreInput,
    *,
    score_type: str = "confidence",
    hallucination_labels: Optional[ScoreInput] = None,
    risk_levels: Sequence[float] = DEFAULT_RISK_LEVELS,
    include_curves: bool = True,
) -> Report:
    """Compute AUROC, AURC, coverage@risk, curves, and commission/omission.

    Args:
        correctness: Binary correctness labels, 1/True means correct.
        scores: Confidence or uncertainty signal.
        score_type: ``"confidence"`` or ``"uncertainty"``.
        hallucination_labels: Optional binary labels where 1 means hallucinated.
            If omitted, wrong answers are used as the hallucination proxy.
        risk_levels: Risk tolerances for coverage@risk, e.g. ``(0.05,)``.
        include_curves: Include per-prefix arrays for plotting.

    Returns:
        Dictionary of scalar metrics plus optional curve arrays.
    """
    corr_bool, score_arr, hall_bool, _ = _prepare_arrays(
        correctness,
        scores,
        hallucination_labels=hallucination_labels,
    )
    n = int(corr_bool.size)
    if n == 0:
        raise ValueError("No valid samples after dropping missing values")

    uncertainty, _, rule = _score_views(score_arr, score_type)
    error = (~corr_bool).astype(int)
    auroc_value = float("nan")
    if np.unique(error).size == 2:
        auroc_value = float(roc_auc_score(error, uncertainty))

    hallucination_auroc = float("nan")
    hall_int = hall_bool.astype(int)
    if np.unique(hall_int).size == 2:
        hallucination_auroc = float(roc_auc_score(hall_int, uncertainty))

    curve = risk_coverage_curve(
        corr_bool.astype(int),
        score_arr,
        score_type=score_type,
        hallucination_labels=hall_int,
    )
    observed_aurc = aurc_from_curve(curve, rate_key="risk")
    observed_h_aurc = aurc_from_curve(curve, rate_key="hallucination_rate")
    ideal = ideal_aurc(corr_bool.astype(int))
    eaurc_value = observed_aurc - ideal
    if np.isfinite(eaurc_value) and eaurc_value < 0 and abs(eaurc_value) < 1e-12:
        eaurc_value = 0.0

    report: Report = {
        "n": n,
        "score_type": score_type,
        "threshold_rule": rule,
        "accuracy": float(corr_bool.mean()),
        "base_risk": float(error.mean()),
        "base_hallucination_rate": float(hall_int.mean()),
        "auroc": _safe_float(auroc_value),
        "hallucination_auroc": _safe_float(hallucination_auroc),
        "aurc": _safe_float(observed_aurc),
        "ideal_aurc": _safe_float(ideal),
        "eaurc": _safe_float(eaurc_value),
        "hallucination_aurc": _safe_float(observed_h_aurc),
        "coverage_at_risk": coverage_at_risk(
            curve, risk_levels=risk_levels, rate_key="risk"
        ),
        "coverage_at_hallucination_risk": coverage_at_risk(
            curve, risk_levels=risk_levels, rate_key="hallucination_rate"
        ),
        "commission_at_full_coverage": int(curve["commission_count"][-1]),
        "omission_at_full_coverage": int(curve["omission_count"][-1]),
    }

    if include_curves:
        report["curves"] = _serialise_curve(curve)
    return report


def evaluate_selective_by_type(
    correctness: ScoreInput,
    scores: ScoreInput,
    question_types: Sequence[Any],
    *,
    score_type: str = "confidence",
    hallucination_labels: Optional[ScoreInput] = None,
    risk_levels: Sequence[float] = DEFAULT_RISK_LEVELS,
    include_all: bool = True,
    include_curves: bool = True,
) -> Dict[str, Report]:
    """Evaluate selective metrics separately for closed/open questions.

    The returned dictionary contains ``"closed"`` and/or ``"open"`` if those
    types are present. ``"all"`` is included by default for convenience.
    """
    corr_bool, score_arr, hall_bool, type_arr = _prepare_arrays(
        correctness,
        scores,
        hallucination_labels=hallucination_labels,
        question_types=question_types,
    )
    if type_arr is None:
        raise ValueError("question_types are required for by-type reporting")

    groups: Dict[str, np.ndarray] = {}
    if include_all:
        groups["all"] = np.ones(type_arr.size, dtype=bool)
    for qtype in ("closed", "open"):
        mask = type_arr == qtype
        if mask.any():
            groups[qtype] = mask

    reports: Dict[str, Report] = {}
    for group_name, mask in groups.items():
        reports[group_name] = evaluate_selective(
            corr_bool[mask].astype(int),
            score_arr[mask],
            score_type=score_type,
            hallucination_labels=hall_bool[mask].astype(int),
            risk_levels=risk_levels,
            include_curves=include_curves,
        )
    return reports


def evaluate_signals_by_type(
    signals: Mapping[str, Union[ScoreInput, Tuple[ScoreInput, str], Mapping[str, Any]]],
    correctness: ScoreInput,
    question_types: Sequence[Any],
    *,
    default_score_type: str = "confidence",
    signal_score_types: Optional[Mapping[str, str]] = None,
    hallucination_labels: Optional[ScoreInput] = None,
    risk_levels: Sequence[float] = DEFAULT_RISK_LEVELS,
    include_all: bool = True,
    include_curves: bool = False,
) -> Dict[str, Dict[str, Report]]:
    """Evaluate multiple confidence/uncertainty signals by question type.

    ``signals`` can be provided in any of these forms:

    - ``{"confidence": conf_array}``
    - ``{"VASE": (vase_array, "uncertainty")}``
    - ``{"VASE": {"scores": vase_array, "score_type": "uncertainty"}}``

    Each signal is evaluated independently and returns an ``all/closed/open``
    report.
    """
    reports: Dict[str, Dict[str, Report]] = {}
    signal_score_types = signal_score_types or {}

    for name, spec in signals.items():
        score_type = signal_score_types.get(name, default_score_type)
        values: ScoreInput

        if isinstance(spec, Mapping):
            if "scores" not in spec:
                raise ValueError(f"Signal {name!r} mapping must contain a 'scores' key")
            values = spec["scores"]
            score_type = str(spec.get("score_type", score_type))
        elif isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[1], str):
            values, score_type = spec
        else:
            values = spec

        reports[name] = evaluate_selective_by_type(
            correctness,
            values,
            question_types,
            score_type=score_type,
            hallucination_labels=hallucination_labels,
            risk_levels=risk_levels,
            include_all=include_all,
            include_curves=include_curves,
        )
    return reports


def extract_arrays_from_results(
    results: Iterable[Any],
    *,
    score_attr: str = "confidence",
    correctness_attr: str = "is_correct",
    question_type_attr: str = "answer_type",
) -> Dict[str, Any]:
    """Extract arrays from result dataclasses or dictionaries.

    This is a convenience bridge for ``SamplingResult``, ``HedgeResult``, or
    JSON-like records. Missing question types are returned as ``None``.
    """

    def get_value(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, Mapping):
            return obj.get(key, default)
        return getattr(obj, key, default)

    scores = []
    correctness = []
    question_types = []
    for item in results:
        scores.append(get_value(item, score_attr))
        correctness.append(get_value(item, correctness_attr))
        question_types.append(get_value(item, question_type_attr, None))

    return {
        "scores": scores,
        "correctness": correctness,
        "question_types": question_types,
    }


__all__ = [
    "DEFAULT_RISK_LEVELS",
    "auroc_error_detection",
    "risk_coverage_curve",
    "aurc_from_curve",
    "ideal_aurc",
    "aurc",
    "coverage_at_risk",
    "commission_omission_curve",
    "evaluate_selective",
    "evaluate_selective_by_type",
    "evaluate_signals_by_type",
    "extract_arrays_from_results",
]
