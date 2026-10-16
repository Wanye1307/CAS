#!/usr/bin/env python3
"""Build paper metric tables from protocol-aligned greedy HAC/CAS/PTrue caches.

This script does not run model inference. It evaluates cached row-level signals
on strict HAC/CAS intersections and writes selective-prediction metric tables
for the paper draft. PTrue rows are filled when protocol-aligned row-level
caches are present under ``03_runs/ptrue/raw``.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_common import BENCH_ROOT, finite, load_jsonl, output_path, row_sample_id, write_json  # noqa: E402
from evaluation.selective import DEFAULT_RISK_LEVELS, evaluate_selective  # noqa: E402
MODEL_LABELS = {
    'qwen3vl_2b': 'Qwen3-VL-2B',
    'qwen3vl_8b': 'Qwen3-VL-8B',
    'internvl3_2b_local': 'InternVL3-2B',
    'internvl3_8b_local': 'InternVL3-8B',
    'lingshu_7b': 'Lingshu-7B',
    'medgemma_4b': 'MedGemma-4B',
    'medvlm_r1': 'MedVLM-R1',
}
DATASET_LABELS = {
    'vqa_rad': 'VQA-RAD',
    'slake': 'SLAKE',
    'vqa_med_2019': 'VQA-Med-2019',
}
MODEL_ORDER = list(MODEL_LABELS)
DATASET_ORDER = ['vqa_rad', 'slake', 'vqa_med_2019']
METHOD_ORDER = [
    'MaxConfidence',
    'NLL',
    'Entropy',
    'Margin',
    'Sampling-SE',
    'PTrue',
    'HAC',
    'HAC-Platt',
    'HAC-Gate',
    'CAS',
]
CAS_ABLATION_ORDER = [
    'CAS-textonly',
    'CAS-black',
    'CAS-noise',
    'CAS-blur',
    'CAS-visual-corrupt',
    'CAS-all',
]
EPS = 1e-12


CLOSED_CAS_ALL = (
    'cas_pred_mean', 'cas_pred_max', 'cas_pred_min',
    'cas_opp_mean', 'cas_opp_max', 'cas_opp_min',
    'cas_cf_support_mean', 'cas_cf_support_max', 'cas_same_pred_frac',
)


def auc(labels: np.ndarray, scores: Sequence[float]) -> float | None:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    mask = np.isfinite(scores)
    labels = labels[mask]
    scores = scores[mask]
    if len(labels) == 0 or len(set(labels.tolist())) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def split_indices(labels: np.ndarray, n_splits: int, seed: int, splitter: str):
    if n_splits <= 1:
        idx = np.arange(len(labels))
        yield idx, idx
        return
    if splitter == 'stratified' and len(set(labels.tolist())) >= 2:
        counts = np.bincount(labels)
        positive_counts = counts[counts > 0]
        if len(positive_counts) >= 2 and int(np.min(positive_counts)) >= n_splits:
            yield from StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(len(labels)), labels)
            return
    yield from KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(len(labels)))


def standardize(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_train = x_train.astype(float, copy=True)
    x_test = x_test.astype(float, copy=True)
    med = np.nanmedian(x_train, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    inds = np.where(~np.isfinite(x_train))
    x_train[inds] = np.take(med, inds[1])
    inds = np.where(~np.isfinite(x_test))
    x_test[inds] = np.take(med, inds[1])
    mu = np.mean(x_train, axis=0)
    sigma = np.std(x_train, axis=0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-8), sigma, 1.0)
    return (x_train - mu) / sigma, (x_test - mu) / sigma


def logreg_cv(labels: np.ndarray, x: np.ndarray, n_splits: int, seed: int, splitter: str) -> dict[str, Any]:
    oof = np.full(len(labels), np.nan, dtype=float)
    folds = []
    fold_aucs = []
    if len(labels) < n_splits or len(set(labels.tolist())) < 2:
        return {'auroc': None, 'auroc_std': None, 'auroc_oof': None, 'scores_oof': oof.tolist(), 'folds': folds}
    for fold, (train_idx, test_idx) in enumerate(split_indices(labels, n_splits, seed, splitter)):
        if len(set(labels[train_idx].tolist())) < 2 or len(set(labels[test_idx].tolist())) < 2:
            continue
        x_train, x_test = standardize(x[train_idx], x[test_idx])
        clf = LogisticRegression(C=1.0, class_weight='balanced', solver='liblinear', max_iter=1000, random_state=seed)
        clf.fit(x_train, labels[train_idx])
        scores = clf.predict_proba(x_test)[:, 1]
        oof[test_idx] = scores
        fold_auc = auc(labels[test_idx], scores)
        if fold_auc is not None:
            fold_aucs.append(fold_auc)
        folds.append({'fold': int(fold), 'train_n': int(len(train_idx)), 'test_n': int(len(test_idx)), 'test_error_n': int(np.sum(labels[test_idx])), 'auroc': fold_auc})
    return {
        'auroc': float(np.mean(fold_aucs)) if fold_aucs else None,
        'auroc_std': float(np.std(fold_aucs)) if fold_aucs else None,
        'auroc_oof': auc(labels, oof),
        'scores_oof': oof.tolist(),
        'folds': folds,
    }


def row_features(rows: Sequence[Mapping[str, Any]], names: Sequence[str]) -> np.ndarray:
    x = np.zeros((len(rows), len(names)), dtype=float)
    for i, row in enumerate(rows):
        feats = row.get('features') or {}
        for j, name in enumerate(names):
            value = feats.get(name)
            x[i, j] = float(value) if finite(value) else np.nan
    return x


def build_intersection(hac_rows: list[dict[str, Any]], cas_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    hac_by_id = {row_sample_id(r): r for r in hac_rows if r.get('status') == 'ok' and isinstance(r.get('is_correct'), bool)}
    cas_by_id = {row_sample_id(r): r for r in cas_rows if r.get('status') == 'ok' and isinstance(r.get('is_correct'), bool)}
    ids = sorted(set(hac_by_id) & set(cas_by_id))
    kept_hac = []
    kept_cas = []
    mismatches = []
    for sid in ids:
        h = hac_by_id[sid]
        c = cas_by_id[sid]
        if bool(h.get('is_correct')) != bool(c.get('is_correct')):
            mismatches.append(sid)
            continue
        kept_hac.append(h)
        kept_cas.append(c)
    if mismatches:
        raise ValueError(f'label mismatch for {len(mismatches)} intersected rows; first={mismatches[:3]}')
    return kept_hac, kept_cas, ids


def _logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=float), 1e-7, 1 - 1e-7)
    return np.log(x / (1 - x))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def fit_hac_platt(c: np.ndarray, y: np.ndarray, h: np.ndarray, lam: float = 0.01, use_logit: bool = True) -> np.ndarray:
    from scipy.optimize import minimize as sp_minimize
    c_in = _logit(c) if use_logit else c
    def nll(params):
        a, b, d = params
        p = np.clip(_sigmoid(a * c_in + b * h + d), 1e-12, 1 - 1e-12)
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        return loss + lam * b ** 2 if lam > 0 else loss
    res = sp_minimize(nll, x0=[1.0, -0.5, 0.0], bounds=[(0, None), (None, 0), (None, None)], method='L-BFGS-B')
    return np.asarray(res.x, dtype=float)


def transform_hac_platt(c: np.ndarray, h: np.ndarray, params: np.ndarray, use_logit: bool = True) -> np.ndarray:
    a, b, d = params
    c_in = _logit(c) if use_logit else c
    return _sigmoid(a * c_in + b * h + d)


def fit_hac_gate(c: np.ndarray, y: np.ndarray, h: np.ndarray, lam: float = 0.01) -> np.ndarray:
    from scipy.optimize import minimize as sp_minimize
    def nll(params):
        a, b = params
        gate = _sigmoid(-a * h + b)
        c_adj = np.clip(c * gate, 1e-8, 1 - 1e-8)
        loss = -np.mean(y * np.log(c_adj) + (1 - y) * np.log(1 - c_adj))
        return loss + lam * a ** 2 if lam > 0 else loss
    res = sp_minimize(nll, x0=[1.0, 0.0], bounds=[(0, None), (None, None)], method='L-BFGS-B')
    return np.asarray(res.x, dtype=float)


def transform_hac_gate(c: np.ndarray, h: np.ndarray, params: np.ndarray) -> np.ndarray:
    a, b = params
    gate = _sigmoid(-a * h + b)
    return np.clip(c * gate, 0, 1)


def hac_calibration_cv(labels: np.ndarray, confidence: np.ndarray, vase: np.ndarray, method: str, n_splits: int, seed: int, splitter: str) -> dict[str, Any]:
    correctness = (labels == 0).astype(float)
    oof = np.full(len(labels), np.nan, dtype=float)
    folds = []
    fold_aucs = []
    if len(labels) < n_splits or len(set(labels.tolist())) < 2:
        return {'auroc': None, 'auroc_std': None, 'auroc_oof': None, 'scores_oof': oof.tolist(), 'folds': folds}
    for fold, (train_idx, test_idx) in enumerate(split_indices(labels, n_splits, seed, splitter)):
        if len(set(labels[train_idx].tolist())) < 2 or len(set(labels[test_idx].tolist())) < 2:
            continue
        c_train = np.asarray(confidence[train_idx], dtype=float)
        c_test = np.asarray(confidence[test_idx], dtype=float)
        h_train = np.asarray(vase[train_idx], dtype=float)
        h_test = np.asarray(vase[test_idx], dtype=float)
        y_train = correctness[train_idx]
        if method == 'hac_platt':
            params = fit_hac_platt(c_train, y_train, h_train, use_logit=True)
            pred_conf = transform_hac_platt(c_test, h_test, params, use_logit=True)
        elif method == 'hac_gate':
            params = fit_hac_gate(c_train, y_train, h_train)
            pred_conf = transform_hac_gate(c_test, h_test, params)
        else:
            raise ValueError(method)
        risk = 1.0 - np.asarray(pred_conf, dtype=float)
        oof[test_idx] = risk
        fold_auc = auc(labels[test_idx], risk)
        if fold_auc is not None:
            fold_aucs.append(fold_auc)
        folds.append({'fold': int(fold), 'train_n': int(len(train_idx)), 'test_n': int(len(test_idx)), 'test_error_n': int(np.sum(labels[test_idx])), 'auroc': fold_auc})
    return {
        'auroc': float(np.mean(fold_aucs)) if fold_aucs else None,
        'auroc_std': float(np.std(fold_aucs)) if fold_aucs else None,
        'auroc_oof': auc(labels, oof),
        'scores_oof': oof.tolist(),
        'folds': folds,
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return '--'
    try:
        f = float(value)
    except (TypeError, ValueError):
        return '--'
    if not math.isfinite(f):
        return '--'
    return f'{f:.{digits}f}'


def fmt_signed(value: Any, digits: int = 3) -> str:
    if value is None:
        return '--'
    try:
        f = float(value)
    except (TypeError, ValueError):
        return '--'
    if not math.isfinite(f):
        return '--'
    return f'{f:+.{digits}f}'


def fmt_ci(low: Any, high: Any, digits: int = 3) -> str:
    if low is None or high is None:
        return '--'
    return f'[{fmt_signed(low, digits)}, {fmt_signed(high, digits)}]'


def esc(value: Any) -> str:
    return str(value).replace('\\', r'\textbackslash{}').replace('_', r'\_').replace('&', r'\&').replace('%', r'\%')


def metric_bundle(error_labels: np.ndarray, risk_scores: Sequence[float] | None) -> dict[str, Any]:
    """Compute the selective.py metric set for an error-risk score.

    ``error_labels`` follows this benchmark's convention: 1 means wrong and 0
    means correct. ``evaluate_selective`` expects correctness labels, so we
    convert here and mark the score as an uncertainty/risk score.
    """
    if risk_scores is None:
        return {
            'n_valid': 0,
            'accuracy': None,
            'base_risk': None,
            'base_hallucination_rate': None,
            'auroc': None,
            'hallucination_auroc': None,
            'aurc': None,
            'ideal_aurc': None,
            'eaurc': None,
            'hallucination_aurc': None,
            'coverage_at_risk_0.01': None,
            'coverage_at_risk_0.05': None,
            'coverage_at_risk_0.10': None,
            'coverage_at_hallucination_risk_0.01': None,
            'coverage_at_hallucination_risk_0.05': None,
            'coverage_at_hallucination_risk_0.10': None,
            'commission_at_full_coverage': None,
            'omission_at_full_coverage': None,
        }
    labels = np.asarray(error_labels, dtype=int)
    scores = np.asarray(risk_scores, dtype=float)
    mask = np.isfinite(scores) & np.isfinite(labels.astype(float))
    labels = labels[mask]
    scores = scores[mask]
    if len(labels) == 0:
        return metric_bundle(labels, None)
    correctness = (labels == 0).astype(int)
    try:
        report = evaluate_selective(
            correctness,
            scores,
            score_type='uncertainty',
            risk_levels=DEFAULT_RISK_LEVELS,
            include_curves=False,
        )
    except ValueError:
        return metric_bundle(labels, None)
    out: dict[str, Any] = {
        'n_valid': int(report.get('n', len(labels))),
        'accuracy': report.get('accuracy'),
        'base_risk': report.get('base_risk'),
        'base_hallucination_rate': report.get('base_hallucination_rate'),
        'auroc': report.get('auroc'),
        'hallucination_auroc': report.get('hallucination_auroc'),
        'aurc': report.get('aurc'),
        'ideal_aurc': report.get('ideal_aurc'),
        'eaurc': report.get('eaurc'),
        'hallucination_aurc': report.get('hallucination_aurc'),
        'commission_at_full_coverage': report.get('commission_at_full_coverage'),
        'omission_at_full_coverage': report.get('omission_at_full_coverage'),
    }
    for level in DEFAULT_RISK_LEVELS:
        key = f'{float(level):.4g}'
        suffix = f'{float(level):.2f}'
        out[f'coverage_at_risk_{suffix}'] = (report.get('coverage_at_risk') or {}).get(key, {}).get('coverage')
        out[f'coverage_at_hallucination_risk_{suffix}'] = (report.get('coverage_at_hallucination_risk') or {}).get(key, {}).get('coverage')
    return out


def paired_bootstrap_delta(labels: np.ndarray, cas_scores: Sequence[float], hac_scores: Sequence[float], n_boot: int, seed: int) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=int)
    cas_scores = np.asarray(cas_scores, dtype=float)
    hac_scores = np.asarray(hac_scores, dtype=float)
    mask = np.isfinite(cas_scores) & np.isfinite(hac_scores)
    labels = labels[mask]
    cas_scores = cas_scores[mask]
    hac_scores = hac_scores[mask]
    if len(labels) < 2 or len(set(labels.tolist())) < 2:
        return {'n': int(len(labels)), 'delta': None, 'ci95_low': None, 'ci95_high': None, 'bootstraps': 0}
    base_cas = auc(labels, cas_scores)
    base_hac = auc(labels, hac_scores)
    if base_cas is None or base_hac is None or n_boot <= 0:
        return {'n': int(len(labels)), 'delta': None, 'ci95_low': None, 'ci95_high': None, 'bootstraps': 0}
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(labels), len(labels))
        if len(set(labels[idx].tolist())) < 2:
            continue
        cas_auc = auc(labels[idx], cas_scores[idx])
        hac_auc = auc(labels[idx], hac_scores[idx])
        if cas_auc is not None and hac_auc is not None:
            deltas.append(cas_auc - hac_auc)
    if not deltas:
        return {'n': int(len(labels)), 'delta': float(base_cas - base_hac), 'ci95_low': None, 'ci95_high': None, 'bootstraps': 0}
    return {
        'n': int(len(labels)),
        'delta': float(base_cas - base_hac),
        'ci95_low': float(np.percentile(deltas, 2.5)),
        'ci95_high': float(np.percentile(deltas, 97.5)),
        'bootstraps': int(len(deltas)),
    }


def make_row(model: str, dataset: str, qtype: str, method: str, labels: np.ndarray, risk: Sequence[float] | None, *, note: str = '') -> dict[str, Any]:
    metrics = metric_bundle(labels, risk)
    row = {
        'model': model,
        'dataset': dataset,
        'task': qtype,
        'method': method,
        'n': int(len(labels)),
        'note': note,
    }
    row.update(metrics)
    return row


def closed_binary_probs(row: Mapping[str, Any]) -> tuple[float, float]:
    original = row.get('original') or {}
    p_yes = float(original.get('p_yes', np.nan))
    p_no = float(original.get('p_no', np.nan))
    return p_yes, p_no


def closed_nll(row: Mapping[str, Any]) -> float:
    target = str(row.get('target_answer') or row.get('target_prediction') or '').lower()
    original = row.get('original') or {}
    if target == 'yes':
        value = original.get('log_p_yes')
        return -float(value) if finite(value) else -math.log(max(float(original.get('p_yes', np.nan)), EPS))
    if target == 'no':
        value = original.get('log_p_no')
        return -float(value) if finite(value) else -math.log(max(float(original.get('p_no', np.nan)), EPS))
    return float('nan')


def binary_entropy(p_yes: float, p_no: float) -> float:
    vals = [p_yes, p_no]
    return float(-sum(max(p, EPS) * math.log(max(p, EPS)) for p in vals if math.isfinite(p)))


def raw_closed_risks(cas_rows: Sequence[Mapping[str, Any]]) -> dict[str, list[float]]:
    maxconf, nll, entropy, margin = [], [], [], []
    for row in cas_rows:
        p_yes, p_no = closed_binary_probs(row)
        max_p = max(p_yes, p_no) if math.isfinite(p_yes) and math.isfinite(p_no) else float('nan')
        maxconf.append(1.0 - max_p if math.isfinite(max_p) else float('nan'))
        nll.append(closed_nll(row))
        entropy.append(binary_entropy(p_yes, p_no))
        margin.append(1.0 - abs(p_yes - p_no) if math.isfinite(p_yes) and math.isfinite(p_no) else float('nan'))
    return {'MaxConfidence': maxconf, 'NLL': nll, 'Entropy': entropy, 'Margin': margin}


def raw_open_risks(cas_rows: Sequence[Mapping[str, Any]]) -> dict[str, list[float] | None]:
    maxconf, nll = [], []
    for row in cas_rows:
        feats = row.get('features') or {}
        conf = float(feats.get('real_confidence', np.nan)) if finite(feats.get('real_confidence')) else float('nan')
        maxconf.append(1.0 - conf if math.isfinite(conf) else float('nan'))
        nll.append(float(feats.get('real_nll', np.nan)) if finite(feats.get('real_nll')) else float('nan'))
    return {'MaxConfidence': maxconf, 'NLL': nll, 'Entropy': None, 'Margin': None}


def hac_risks(hac_rows: Sequence[Mapping[str, Any]]) -> dict[str, list[float]]:
    hac, se = [], []
    for row in hac_rows:
        conf = float(row.get('hac_confidence_n10', np.nan)) if finite(row.get('hac_confidence_n10')) else float('nan')
        hac.append(1.0 - conf if math.isfinite(conf) else float('nan'))
        se.append(float(row.get('SE', np.nan)) if finite(row.get('SE')) else float('nan'))
    return {'HAC': hac, 'Sampling-SE': se}


def ptrue_risks(path: Path, reference_rows: Sequence[Mapping[str, Any]]) -> tuple[list[float] | None, str]:
    if not path.exists():
        return None, 'missing protocol-aligned PTrue cache'
    by_id = {
        row_sample_id(row): row
        for row in load_jsonl(path)
        if row.get('status') == 'ok'
    }
    risks: list[float] = []
    missing = 0
    mismatches = []
    for ref in reference_rows:
        sid = row_sample_id(ref)
        row = by_id.get(sid)
        if row is None:
            missing += 1
            risks.append(float('nan'))
            continue
        if isinstance(row.get('is_correct'), bool) and bool(row.get('is_correct')) != bool(ref.get('is_correct')):
            mismatches.append(sid)
            risks.append(float('nan'))
            continue
        value = row.get('p_true', row.get('confidence'))
        risk = 1.0 - float(value) if finite(value) else float('nan')
        risks.append(float(risk))
    if mismatches:
        raise ValueError(f'PTrue label mismatch for {len(mismatches)} rows in {path}; first={mismatches[:3]}')
    note = f'missing rows: {missing}' if missing else ''
    return risks, note


def cas_subset_matrix(rows: Sequence[Mapping[str, Any]], subset: Sequence[str]) -> np.ndarray:
    names = []
    for base in subset:
        names.append(f'cas_pred_{base}')
        names.append(f'cas_opp_{base}')
    x = np.full((len(rows), len(names) + 2), np.nan, dtype=float)
    for i, row in enumerate(rows):
        feats = row.get('features') or {}
        values = []
        for name in names:
            value = feats.get(name)
            values.append(float(value) if finite(value) else np.nan)
        supports = []
        same_pred = []
        target = str(row.get('target_answer') or row.get('target_prediction') or '').lower()
        for base in subset:
            raw_name = 'textonly' if base == 'text' else base
            cf = (row.get('baselines') or {}).get(raw_name) or {}
            if target == 'yes':
                support = cf.get('p_yes')
            elif target == 'no':
                support = cf.get('p_no')
            else:
                support = np.nan
            supports.append(float(support) if finite(support) else np.nan)
            p_yes = float(cf.get('p_yes', np.nan)) if finite(cf.get('p_yes')) else np.nan
            p_no = float(cf.get('p_no', np.nan)) if finite(cf.get('p_no')) else np.nan
            if math.isfinite(p_yes) and math.isfinite(p_no):
                cf_pred = 'yes' if p_yes >= p_no else 'no'
                same_pred.append(1.0 if cf_pred == target else 0.0)
        values.append(float(np.nanmean(supports)) if supports else np.nan)
        values.append(float(np.nanmean(same_pred)) if same_pred else np.nan)
        x[i, :] = np.asarray(values, dtype=float)
    return x


def setting_files(args: argparse.Namespace) -> list[tuple[str, str, str, Path, Path]]:
    out = []
    for qtype, cas_dir in [('closed', Path(args.cas_closed_dir)), ('open', Path(args.cas_open_dir))]:
        if args.qtypes and qtype not in args.qtypes:
            continue
        for hac_file in sorted(Path(args.hac_dir).glob(f'*__{qtype}.jsonl')):
            parts = hac_file.stem.split('__')
            if len(parts) != 3:
                continue
            model, dataset, task = parts
            if args.models and model not in args.models:
                continue
            if model not in MODEL_LABELS or dataset not in DATASET_LABELS:
                continue
            cas_file = output_path(cas_dir, model, dataset, task)
            if not cas_file.exists():
                continue
            out.append((model, dataset, task, hac_file, cas_file))
    return out


def evaluate_setting(model: str, dataset: str, qtype: str, hac_file: Path, cas_file: Path, args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    hac_rows, cas_rows, _ids = build_intersection(load_jsonl(hac_file), load_jsonl(cas_file))
    labels = np.asarray([0 if bool(r['is_correct']) else 1 for r in hac_rows], dtype=int)
    rows: list[dict[str, Any]] = []
    ablations: list[dict[str, Any]] = []
    significance: dict[str, Any] | None = None

    raw = raw_closed_risks(cas_rows) if qtype == 'closed' else raw_open_risks(cas_rows)
    all_raw: dict[str, Any] = {**raw, **hac_risks(hac_rows)}
    hac_candidates: dict[str, dict[str, Any]] = {}
    for method in METHOD_ORDER:
        if method in {'HAC-Platt', 'HAC-Gate', 'CAS'}:
            continue
        if method == 'PTrue':
            ptrue_file = output_path(Path(args.ptrue_dir), model, dataset, qtype)
            risks, note = ptrue_risks(ptrue_file, hac_rows)
            rows.append(make_row(model, dataset, qtype, method, labels, risks, note=note))
        else:
            method_row = make_row(model, dataset, qtype, method, labels, all_raw.get(method))
            rows.append(method_row)
            if qtype == 'closed' and method == 'HAC':
                hac_candidates[method] = {'row': method_row, 'scores': all_raw.get(method)}

    confidence = np.asarray([float(r.get('hac_confidence_n10')) for r in hac_rows], dtype=float)
    vase = np.asarray([float(r.get('VASE')) for r in hac_rows], dtype=float)
    for name, cal_method in [('HAC-Platt', 'hac_platt'), ('HAC-Gate', 'hac_gate')]:
        rep = hac_calibration_cv(labels, confidence, vase, cal_method, args.outer_splits, args.seed, args.splitter)
        method_row = make_row(model, dataset, qtype, name, labels, rep.get('scores_oof'))
        rows.append(method_row)
        if qtype == 'closed':
            hac_candidates[name] = {'row': method_row, 'scores': rep.get('scores_oof')}

    if qtype == 'closed':
        cas_rep = logreg_cv(labels, row_features(cas_rows, CLOSED_CAS_ALL), args.outer_splits, args.seed, args.splitter)
        cas_row = make_row(model, dataset, qtype, 'CAS', labels, cas_rep.get('scores_oof'))
        rows.append(cas_row)
        valid_hac = [
            (name, candidate)
            for name, candidate in hac_candidates.items()
            if candidate.get('row', {}).get('auroc') is not None
        ]
        if valid_hac and cas_row.get('auroc') is not None:
            best_name, best = max(valid_hac, key=lambda item: float(item[1]['row']['auroc']))
            boot = paired_bootstrap_delta(labels, cas_rep.get('scores_oof'), best.get('scores'), args.bootstraps, args.seed)
            significance = {
                'model': model,
                'dataset': dataset,
                'task': qtype,
                'n': int(len(labels)),
                'n_error': int(np.sum(labels == 1)),
                'accuracy': float(np.mean(labels == 0)) if len(labels) else None,
                'hac_best_method': best_name,
                'hac_best_auroc': best['row'].get('auroc'),
                'cas_auroc': cas_row.get('auroc'),
                'delta_auroc': float(cas_row['auroc'] - best['row']['auroc']),
                'bootstrap_delta': boot.get('delta'),
                'ci95_low': boot.get('ci95_low'),
                'ci95_high': boot.get('ci95_high'),
                'bootstraps': boot.get('bootstraps'),
                'significant_positive': bool(boot.get('ci95_low') is not None and float(boot['ci95_low']) > 0.0),
            }
        specs = {
            'CAS-textonly': ('text',),
            'CAS-black': ('black',),
            'CAS-noise': ('noise',),
            'CAS-blur': ('blur',),
            'CAS-visual-corrupt': ('black', 'noise', 'blur'),
            'CAS-all': ('text', 'black', 'noise', 'blur'),
        }
        for name in CAS_ABLATION_ORDER:
            rep = logreg_cv(labels, cas_subset_matrix(cas_rows, specs[name]), args.outer_splits, args.seed, args.splitter)
            row = make_row(model, dataset, qtype, name, labels, rep.get('scores_oof'))
            row['features'] = list(specs[name])
            ablations.append(row)
    else:
        # Open-ended CAS is intentionally excluded from the paper tables.
        # The open rows remain useful for HAC and likelihood/sampling baselines,
        # but CAS claims are restricted to closed-ended VQA.
        pass
    return rows, ablations, significance


def sort_key(row: Mapping[str, Any]) -> tuple[int, int, int, int]:
    return (
        MODEL_ORDER.index(row['model']) if row['model'] in MODEL_ORDER else 99,
        DATASET_ORDER.index(row['dataset']) if row['dataset'] in DATASET_ORDER else 99,
        0 if row['task'] == 'closed' else 1,
        METHOD_ORDER.index(row['method']) if row['method'] in METHOD_ORDER else 99,
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'model', 'dataset', 'task', 'method', 'n', 'n_valid', 'accuracy', 'base_risk',
        'base_hallucination_rate', 'auroc', 'hallucination_auroc', 'aurc', 'ideal_aurc',
        'eaurc', 'hallucination_aurc', 'coverage_at_risk_0.01', 'coverage_at_risk_0.05',
        'coverage_at_risk_0.10', 'coverage_at_hallucination_risk_0.01',
        'coverage_at_hallucination_risk_0.05', 'coverage_at_hallucination_risk_0.10',
        'commission_at_full_coverage', 'omission_at_full_coverage', 'note',
    ]
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def write_significance_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'model', 'dataset', 'task', 'n', 'n_error', 'accuracy', 'hac_best_method',
        'hac_best_auroc', 'cas_auroc', 'delta_auroc', 'bootstrap_delta',
        'ci95_low', 'ci95_high', 'bootstraps', 'significant_positive',
    ]
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def mean_field(members: Sequence[Mapping[str, Any]], field: str) -> float | None:
    vals = []
    for row in members:
        value = row.get(field)
        try:
            f = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            vals.append(f)
    return float(np.mean(vals)) if vals else None


SUMMARY_FIELDS = [
    'accuracy', 'base_risk', 'auroc', 'hallucination_auroc', 'aurc', 'ideal_aurc',
    'eaurc', 'hallucination_aurc', 'coverage_at_risk_0.01', 'coverage_at_risk_0.05',
    'coverage_at_risk_0.10', 'coverage_at_hallucination_risk_0.05',
]


def aggregate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    present_tasks = ['closed', 'open']
    for row in rows:
        if row.get('n_valid', 0) == 0:
            continue
        groups.setdefault((row['task'], row['method']), []).append(row)
    out = []
    for task in present_tasks:
        methods = METHOD_ORDER if task == 'closed' else [m for m in METHOD_ORDER if m != 'CAS']
        for method in methods:
            members = groups.get((task, method), [])
            row: dict[str, Any] = {'task': task, 'method': method, 'settings': len(members)}
            for field in SUMMARY_FIELDS:
                row[f'mean_{field}'] = mean_field(members, field)
            out.append(row)
    return out


def aggregate_ablation(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get('n_valid', 0) == 0:
            continue
        groups.setdefault(row['method'], []).append(row)
    out = []
    for method, members in groups.items():
        row: dict[str, Any] = {'method': method, 'settings': len(members)}
        for field in SUMMARY_FIELDS:
            row[f'mean_{field}'] = mean_field(members, field)
        out.append(row)
    out.sort(key=lambda r: CAS_ABLATION_ORDER.index(r['method']) if r['method'] in CAS_ABLATION_ORDER else 99)
    return out


def latex_summary(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = ["""\\begin{table*}[t]
\\centering
\\caption{Benchmark summary by task and method under the shared greedy-label protocol. Metrics follow \\texttt{evaluation/selective.py}; AUROC is error-detection AUROC, AURC/E-AURC are selective-risk areas, and Cov@r is maximum coverage at selective risk $\\leq r$.}
\\label{tab:benchmark_main_summary}
\\small
\\begin{tabular}{llrrrrrrrr}
\\toprule
Task & Method & Settings & Acc. & Base risk & AUROC $\\uparrow$ & AURC $\\downarrow$ & E-AURC $\\downarrow$ & Cov@5\\% $\\uparrow$ & Cov@10\\% $\\uparrow$ \\\\
\\midrule
"""]
    for row in rows:
        lines.append(
            f"{esc(row['task'])} & {esc(row['method'])} & {row['settings']} & "
            f"{fmt(row.get('mean_accuracy'))} & {fmt(row.get('mean_base_risk'))} & "
            f"{fmt(row.get('mean_auroc'))} & {fmt(row.get('mean_aurc'))} & {fmt(row.get('mean_eaurc'))} & "
            f"{fmt(row.get('mean_coverage_at_risk_0.05'))} & {fmt(row.get('mean_coverage_at_risk_0.10'))} " + r"\\" + "\n"
        )
    lines.append("""\\bottomrule
\\end{tabular}
\\end{table*}
""")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(lines), encoding='utf-8')


def latex_selective_ops(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = ["""\\begin{table*}[t]
\\centering
\\caption{Selective operating points from \\texttt{evaluation/selective.py}. Cov@r is the largest answer coverage whose selective risk is at most $r$. Hallucination metrics use wrong answers as the hallucination proxy in this benchmark.}
\\label{tab:benchmark_selective_ops}
\\small
\\begin{tabular}{llrrrrrr}
\\toprule
Task & Method & Settings & Ideal AURC $\\downarrow$ & H-AUROC $\\uparrow$ & H-AURC $\\downarrow$ & Cov@1\\% $\\uparrow$ & HCov@5\\% $\\uparrow$ \\\\
\\midrule
"""]
    for row in rows:
        lines.append(
            f"{esc(row['task'])} & {esc(row['method'])} & {row['settings']} & "
            f"{fmt(row.get('mean_ideal_aurc'))} & {fmt(row.get('mean_hallucination_auroc'))} & "
            f"{fmt(row.get('mean_hallucination_aurc'))} & {fmt(row.get('mean_coverage_at_risk_0.01'))} & "
            f"{fmt(row.get('mean_coverage_at_hallucination_risk_0.05'))} " + r"\\" + "\n"
        )
    lines.append("""\\bottomrule
\\end{tabular}
\\end{table*}
""")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(lines), encoding='utf-8')


def latex_ablation(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = ["""\\begin{table}[t]
\\centering
\\caption{Closed-ended CAS counterfactual ablation. Metrics are averaged over 21 closed model-dataset settings.}
\\label{tab:cas_closed_ablation}
\\small
\\begin{tabular}{lrrrrrr}
\\toprule
CAS variant & Settings & AUROC $\\uparrow$ & AURC $\\downarrow$ & E-AURC $\\downarrow$ & Cov@5\\% $\\uparrow$ & Cov@10\\% $\\uparrow$ \\\\
\\midrule
"""]
    for row in rows:
        lines.append(
            f"{esc(row['method'])} & {row['settings']} & {fmt(row.get('mean_auroc'))} & "
            f"{fmt(row.get('mean_aurc'))} & {fmt(row.get('mean_eaurc'))} & "
            f"{fmt(row.get('mean_coverage_at_risk_0.05'))} & {fmt(row.get('mean_coverage_at_risk_0.10'))} " + r"\\" + "\n"
        )
    lines.append("""\\bottomrule
\\end{tabular}
\\end{table}
""")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(lines), encoding='utf-8')


def dataset_table_label(dataset: str) -> str:
    return f"tab:benchmark_main_full_{dataset.replace('_', '')}"


def bootstrap_mean_ci(values: Sequence[float], n_boot: int, seed: int) -> tuple[float | None, float | None]:
    vals = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    if len(vals) == 0 or n_boot <= 0:
        return None, None
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(vals), len(vals))
        means.append(float(np.mean(vals[idx])))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def latex_headline(path: Path, rows: Sequence[Mapping[str, Any]], n_boot: int, seed: int) -> None:
    rows = [row for row in rows if row.get('delta_auroc') is not None]
    wins = sum(1 for row in rows if float(row.get('delta_auroc', 0.0)) > 0.0)
    significant = sum(1 for row in rows if row.get('significant_positive'))
    deltas = [float(row['delta_auroc']) for row in rows]
    mean_delta = float(np.mean(deltas)) if deltas else None
    ci_low, ci_high = bootstrap_mean_ci(deltas, n_boot, seed)
    lines = ["""\\begin{table}[t]
\\centering
\\caption{Headline closed-ended error-detection AUROC comparison under the shared greedy-label protocol. HAC is the best HAC-family score in each setting. CI is a bootstrap interval over model--dataset settings.}
\\label{tab:headline_summary}
\\small
\\begin{tabular}{llrrrr}
\\toprule
Question type & CAS row & Wins & Sig. wins & Mean $\\Delta$ AUROC & 95\\% CI \\\\
\\midrule
"""]
    lines.append(
        f"Closed & CAS-only & {wins}/{len(rows)} & {significant}/{len(rows)} & "
        f"{fmt_signed(mean_delta)} & {fmt_ci(ci_low, ci_high)} " + r"\\" + "\n"
    )
    lines.append("""\\bottomrule
\\end{tabular}
\\end{table}
""")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(lines), encoding='utf-8')


def latex_closed_results(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = ["""\\begin{table*}[t]
\\centering
\\caption{Closed-ended HAC versus CAS results with paired bootstrap confidence intervals. AUROC is error-detection AUROC; HAC best is selected within HAC/HAC-Platt/HAC-Gate under the same cross-validation protocol.}
\\label{tab:closed_results}
\\small
\\begin{tabular}{llrrrrrr}
\\toprule
Model & Dataset & $n$ & Acc. & HAC best & CAS & $\\Delta$ & 95\\% CI \\\\
\\midrule
"""]
    for row in rows:
        lines.append(
            f"{esc(MODEL_LABELS.get(row['model'], row['model']))} & {esc(DATASET_LABELS.get(row['dataset'], row['dataset']))} & "
            f"{row.get('n', 0)} & {fmt(row.get('accuracy'))} & "
            f"{fmt(row.get('hac_best_auroc'))} & {fmt(row.get('cas_auroc'))} & "
            f"{fmt_signed(row.get('delta_auroc'))} & {fmt_ci(row.get('ci95_low'), row.get('ci95_high'))} " + r"\\" + "\n"
        )
    lines.append("""\\bottomrule
\\end{tabular}
\\end{table*}
""")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(lines), encoding='utf-8')

def latex_full(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines: list[str] = []
    for dataset in DATASET_ORDER:
        dataset_rows = [row for row in rows if row['dataset'] == dataset]
        if not dataset_rows:
            continue
        dataset_label = esc(DATASET_LABELS.get(dataset, dataset))
        label = dataset_table_label(dataset)
        lines.append(rf"""\begin{{longtable}}{{lllrrrrrr}}
\caption{{Full benchmark table on {dataset_label}: model $\times$ task $\times$ method under the shared greedy-label protocol. PTrue uses protocol-aligned self-evaluation caches when available.}}\label{{{label}}}\\
\toprule
Model & Task & Method & $n$ & Acc. & AUROC $\uparrow$ & AURC $\downarrow$ & E-AURC $\downarrow$ & Cov@10\% $\uparrow$ \\
\midrule
\endfirsthead
\toprule
Model & Task & Method & $n$ & Acc. & AUROC $\uparrow$ & AURC $\downarrow$ & E-AURC $\downarrow$ & Cov@10\% $\uparrow$ \\
\midrule
\endhead
""")
        for row in dataset_rows:
            lines.append(
                f"{esc(MODEL_LABELS.get(row['model'], row['model']))} & "
                f"{esc(row['task'])} & {esc(row['method'])} & {row['n_valid']} & "
                f"{fmt(row.get('accuracy'))} & {fmt(row.get('auroc'))} & {fmt(row.get('aurc'))} & "
                f"{fmt(row.get('eaurc'))} & {fmt(row.get('coverage_at_risk_0.10'))} " + r"\\" + "\n"
            )
        lines.append(r"""\bottomrule
\end{longtable}

""")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(lines), encoding='utf-8')

def run(args: argparse.Namespace) -> None:
    all_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    significance_rows: list[dict[str, Any]] = []
    for model, dataset, qtype, hac_file, cas_file in setting_files(args):
        rows, ablations, significance = evaluate_setting(model, dataset, qtype, hac_file, cas_file, args)
        all_rows.extend(rows)
        ablation_rows.extend(ablations)
        if significance is not None:
            significance_rows.append(significance)
        print(f'evaluated {model}/{dataset}/{qtype}: methods={len(rows)} ablations={len(ablations)}')
    all_rows.sort(key=sort_key)
    ablation_rows.sort(key=lambda r: (MODEL_ORDER.index(r['model']) if r['model'] in MODEL_ORDER else 99, DATASET_ORDER.index(r['dataset']) if r['dataset'] in DATASET_ORDER else 99, CAS_ABLATION_ORDER.index(r['method']) if r['method'] in CAS_ABLATION_ORDER else 99))
    significance_rows.sort(key=lambda r: (MODEL_ORDER.index(r['model']) if r['model'] in MODEL_ORDER else 99, DATASET_ORDER.index(r['dataset']) if r['dataset'] in DATASET_ORDER else 99))

    out_dir = Path(args.output_dir)
    table_dir = Path(args.table_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    summary = aggregate(all_rows)
    ablation_summary = aggregate_ablation(ablation_rows)

    write_json(out_dir / 'benchmark_main_metrics.json', {'rows': all_rows, 'summary': summary, 'closed_significance': significance_rows})
    write_json(out_dir / 'cas_closed_ablation.json', {'rows': ablation_rows, 'summary': ablation_summary})
    write_json(out_dir / 'closed_significance.json', {'rows': significance_rows})
    write_csv(out_dir / 'benchmark_main_metrics.csv', all_rows)
    write_csv(out_dir / 'cas_closed_ablation.csv', ablation_rows)
    write_significance_csv(out_dir / 'closed_significance.csv', significance_rows)
    latex_summary(table_dir / 'benchmark_main_summary.tex', summary)
    latex_selective_ops(table_dir / 'benchmark_selective_ops.tex', summary)
    latex_ablation(table_dir / 'cas_closed_ablation.tex', ablation_summary)
    latex_headline(table_dir / 'headline_summary.tex', significance_rows, args.bootstraps, args.seed)
    latex_closed_results(table_dir / 'closed_results.tex', significance_rows)
    latex_full(table_dir / 'benchmark_main_full.tex', all_rows)
    print(f'wrote {out_dir}')
    print(f'wrote latex tables to {table_dir}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hac-dir', default=str(BENCH_ROOT / '03_runs/hac/raw'))
    parser.add_argument('--cas-closed-dir', default=str(BENCH_ROOT / '03_runs/cas_closed/features'))
    parser.add_argument('--cas-open-dir', default=str(BENCH_ROOT / '03_runs/cas_open/features'))
    parser.add_argument('--models', nargs='+', help='Restrict to named model keys')
    parser.add_argument('--qtypes', nargs='+', choices=['closed', 'open'], help='Restrict question types')
    parser.add_argument('--ptrue-dir', default=str(BENCH_ROOT / '03_runs/ptrue/raw'))
    parser.add_argument('--output-dir', default=str(BENCH_ROOT / '04_results/paper_metrics'))
    parser.add_argument('--table-dir', default=str(BENCH_ROOT / '08_overleaf_draft_local/tables'))
    parser.add_argument('--outer-splits', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--splitter', choices=['kfold', 'stratified'], default='stratified')
    parser.add_argument('--bootstraps', type=int, default=2000)
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())
