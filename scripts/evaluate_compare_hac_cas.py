#!/usr/bin/env python3
"""Evaluate HAC and CAS on strict greedy-label sample intersections."""

from __future__ import annotations

import argparse
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
from medvlm import CalibrationPipeline  # noqa: E402

CLOSED_CAS_ALL = (
    'cas_pred_mean', 'cas_pred_max', 'cas_pred_min',
    'cas_opp_mean', 'cas_opp_max', 'cas_opp_min',
    'cas_cf_support_mean', 'cas_cf_support_max', 'cas_same_pred_frac',
)
OPEN_CAS_ONLY = ('cas_text', 'cas_black', 'cas_noise', 'cas_blur', 'cas_mean', 'cas_max')
OPEN_CAS_FULL = (
    'real_nll', 'real_confidence', 'cf_logprob_max', 'cf_logprob_mean',
    'cf_confidence_max', 'cf_confidence_mean',
    'cas_text', 'cas_black', 'cas_noise', 'cas_blur', 'cas_mean', 'cas_max',
    'cas_positive_mean', 'cas_positive_max', 'token_count_log',
)

def auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
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


def fixed_cv(labels: np.ndarray, scores: np.ndarray, n_splits: int, seed: int, splitter: str) -> dict[str, Any]:
    oof = np.asarray(scores, dtype=float).copy()
    folds = []
    fold_aucs = []
    if len(labels) >= n_splits:
        for fold, (_train_idx, test_idx) in enumerate(split_indices(labels, n_splits, seed, splitter)):
            fold_auc = auc(labels[test_idx], oof[test_idx])
            if fold_auc is not None:
                fold_aucs.append(fold_auc)
            folds.append({'fold': int(fold), 'test_n': int(len(test_idx)), 'test_error_n': int(np.sum(labels[test_idx])), 'auroc': fold_auc})
    return {
        'auroc': float(np.mean(fold_aucs)) if fold_aucs else auc(labels, oof),
        'auroc_std': float(np.std(fold_aucs)) if fold_aucs else None,
        'auroc_oof': auc(labels, oof),
        'scores_oof': oof.tolist(),
        'folds': folds,
    }


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


def hac_calibration_cv(labels: np.ndarray, confidence: np.ndarray, vase: np.ndarray, method: str, n_splits: int, seed: int, splitter: str) -> dict[str, Any]:
    correctness = labels == 0
    oof_conf = np.full(len(labels), np.nan, dtype=float)
    folds = []
    fold_aucs = []
    if len(labels) < n_splits or len(set(labels.tolist())) < 2:
        return {'auroc': None, 'auroc_std': None, 'auroc_oof': None, 'scores_oof': oof_conf.tolist(), 'folds': folds}
    for fold, (train_idx, test_idx) in enumerate(split_indices(labels, n_splits, seed, splitter)):
        if len(set(labels[train_idx].tolist())) < 2 or len(set(labels[test_idx].tolist())) < 2:
            continue
        cal = CalibrationPipeline(method=method)
        cal.fit(confidence[train_idx], correctness[train_idx], hallucination_scores=vase[train_idx])
        pred_conf = cal.transform(confidence[test_idx], hallucination_scores=vase[test_idx])
        risk = 1.0 - np.asarray(pred_conf, dtype=float)
        oof_conf[test_idx] = risk
        fold_auc = auc(labels[test_idx], risk)
        if fold_auc is not None:
            fold_aucs.append(fold_auc)
        folds.append({'fold': int(fold), 'train_n': int(len(train_idx)), 'test_n': int(len(test_idx)), 'test_error_n': int(np.sum(labels[test_idx])), 'auroc': fold_auc})
    return {
        'auroc': float(np.mean(fold_aucs)) if fold_aucs else None,
        'auroc_std': float(np.std(fold_aucs)) if fold_aucs else None,
        'auroc_oof': auc(labels, oof_conf),
        'scores_oof': oof_conf.tolist(),
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


def paired_bootstrap(labels: np.ndarray, cas_scores: Sequence[float], hac_scores: Sequence[float], n_boot: int, seed: int) -> dict[str, Any]:
    cas_scores = np.asarray(cas_scores, dtype=float)
    hac_scores = np.asarray(hac_scores, dtype=float)
    mask = np.isfinite(cas_scores) & np.isfinite(hac_scores)
    labels = labels[mask]
    cas_scores = cas_scores[mask]
    hac_scores = hac_scores[mask]
    if len(labels) < 2 or len(set(labels.tolist())) < 2 or n_boot <= 0:
        return {'n': int(len(labels)), 'delta': None, 'ci95': [None, None], 'bootstraps': 0}
    base_cas = auc(labels, cas_scores)
    base_hac = auc(labels, hac_scores)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(labels), len(labels))
        if len(set(labels[idx].tolist())) < 2:
            continue
        a = auc(labels[idx], cas_scores[idx])
        b = auc(labels[idx], hac_scores[idx])
        if a is not None and b is not None:
            deltas.append(a - b)
    if not deltas or base_cas is None or base_hac is None:
        return {'n': int(len(labels)), 'delta': None, 'ci95': [None, None], 'bootstraps': len(deltas)}
    return {
        'n': int(len(labels)),
        'delta': float(base_cas - base_hac),
        'ci95': [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
        'bootstraps': int(len(deltas)),
    }


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


def evaluate_setting(hac_file: Path, cas_file: Path, qtype: str, args: argparse.Namespace) -> dict[str, Any]:
    hac_rows, cas_rows, ids = build_intersection(load_jsonl(hac_file), load_jsonl(cas_file))
    labels = np.asarray([0 if bool(r['is_correct']) else 1 for r in hac_rows], dtype=int)
    confidence = np.asarray([float(r.get('hac_confidence_n10')) for r in hac_rows], dtype=float)
    vase = np.asarray([float(r.get('VASE')) for r in hac_rows], dtype=float)
    result: dict[str, Any] = {
        'qtype': qtype,
        'hac_file': str(hac_file),
        'cas_file': str(cas_file),
        'n_intersection': int(len(ids)),
        'n_error': int(np.sum(labels == 1)) if len(labels) else 0,
        'accuracy': float(np.mean(labels == 0)) if len(labels) else None,
        'scores': {},
    }
    result['scores']['HAC-confidence-N10'] = fixed_cv(labels, 1.0 - confidence, args.outer_splits, args.seed, args.splitter)
    result['scores']['VASE'] = fixed_cv(labels, vase, args.outer_splits, args.seed, args.splitter)
    result['scores']['HAC-Platt'] = hac_calibration_cv(labels, confidence, vase, 'hac_platt', args.outer_splits, args.seed, args.splitter)
    result['scores']['HAC-Gate'] = hac_calibration_cv(labels, confidence, vase, 'hac_gate', args.outer_splits, args.seed, args.splitter)
    hac_candidates = {k: v for k, v in result['scores'].items() if k in {'HAC-Platt', 'HAC-Gate'}}
    hac_best_name = max(hac_candidates, key=lambda k: float(hac_candidates[k]['auroc'] or -1.0)) if hac_candidates else 'HAC-Platt'
    result['hac_best'] = {'name': hac_best_name, **hac_candidates[hac_best_name]}

    result['cas_primary'] = None
    result['delta_primary_vs_hac_best'] = None
    result['paired_bootstrap'] = None
    if qtype == 'closed':
        cas_features = CLOSED_CAS_ALL
        cas_name = 'CAS-LogReg-cas_all'
        x = row_features(cas_rows, cas_features)
        result['scores'][cas_name] = logreg_cv(labels, x, args.outer_splits, args.seed, args.splitter)
        result['cas_primary'] = {'name': cas_name, 'features': list(cas_features), **result['scores'][cas_name]}
        cas_auc = result['cas_primary'].get('auroc')
        hac_auc = result['hac_best'].get('auroc')
        if cas_auc is not None and hac_auc is not None:
            result['delta_primary_vs_hac_best'] = float(cas_auc - hac_auc)
        result['paired_bootstrap'] = paired_bootstrap(
            labels,
            result['cas_primary']['scores_oof'],
            result['hac_best']['scores_oof'],
            args.bootstraps,
            args.seed,
        )
    elif qtype == 'open':
        cas_name = 'CAS-LogReg-cas_only'
        x = row_features(cas_rows, OPEN_CAS_ONLY)
        result['scores'][cas_name] = logreg_cv(labels, x, args.outer_splits, args.seed, args.splitter)
        result['cas_primary'] = {'name': cas_name, 'features': list(OPEN_CAS_ONLY), **result['scores'][cas_name]}
        cas_auc = result['cas_primary'].get('auroc')
        hac_auc = result['hac_best'].get('auroc')
        if cas_auc is not None and hac_auc is not None:
            result['delta_primary_vs_hac_best'] = float(cas_auc - hac_auc)
        result['paired_bootstrap'] = paired_bootstrap(
            labels, result['cas_primary']['scores_oof'],
            result['hac_best']['scores_oof'], args.bootstraps, args.seed,
        )
        if args.include_cas_full:
            full_name = 'CAS+Likelihood-LogReg-full'
            result['scores'][full_name] = logreg_cv(
                labels, row_features(cas_rows, OPEN_CAS_FULL),
                args.outer_splits, args.seed, args.splitter,
            )
    return result


def write_markdown(path: Path, result: Mapping[str, Any]) -> None:
    def fmt(value: Any) -> str:
        try:
            f = float(value)
        except (TypeError, ValueError):
            return 'NA'
        return f'{f:.4f}' if math.isfinite(f) else 'NA'
    lines = [
        '# HAC vs CAS Primary Comparison',
        '',
        f'- qtype: `{result.get("qtype")}`',
        f'- n intersection: {result.get("n_intersection")}',
        f'- errors: {result.get("n_error")}',
        f'- accuracy: {fmt(result.get("accuracy"))}',
        '',
        '| method | AUROC mean | AUROC OOF | std |',
        '|---|---:|---:|---:|',
    ]
    for name, report in result.get('scores', {}).items():
        lines.append(f'| {name} | {fmt(report.get("auroc"))} | {fmt(report.get("auroc_oof"))} | {fmt(report.get("auroc_std"))} |')
    boot = result.get('paired_bootstrap') or {}
    cas_primary = result.get('cas_primary') or {}
    lines.extend([
        '',
        f'- HAC-best: `{result.get("hac_best", {}).get("name")}`',
        f'- CAS-primary: `{cas_primary.get("name")}`',
        f'- delta primary vs HAC-best: {fmt(result.get("delta_primary_vs_hac_best"))}',
        f'- paired bootstrap delta: {fmt(boot.get("delta"))}, CI95=[{fmt((boot.get("ci95") or [None, None])[0])}, {fmt((boot.get("ci95") or [None, None])[1])}], B={boot.get("bootstraps")}',
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run(args: argparse.Namespace) -> None:
    if args.hac_file and args.cas_file:
        result = evaluate_setting(Path(args.hac_file), Path(args.cas_file), args.qtype, args)
        write_json(Path(args.output_json), result)
        write_markdown(Path(args.output_md), result)
        print(f'wrote {args.output_json}')
        print(f'wrote {args.output_md}')
        return
    hac_dir = Path(args.hac_dir)
    cas_dir = Path(args.cas_dir)
    out_dir = Path(args.output_dir)
    summaries = []
    for hac_file in sorted(hac_dir.glob('*.jsonl')):
        parts = hac_file.stem.split('__')
        if len(parts) != 3:
            continue
        model, dataset, qtype = parts
        cas_file = output_path(cas_dir, model, dataset, qtype)
        if not cas_file.exists():
            continue
        result = evaluate_setting(hac_file, cas_file, qtype, args)
        result.update({'model': model, 'dataset': dataset})
        out_json = out_dir / 'json' / f'{model}__{dataset}__{qtype}.json'
        out_md = out_dir / 'tables' / f'{model}__{dataset}__{qtype}.md'
        write_json(out_json, result)
        write_markdown(out_md, result)
        summaries.append(result)
        print(f'evaluated {model}/{dataset}/{qtype}: delta={result.get("delta_primary_vs_hac_best")}')
    write_json(out_dir / 'json' / 'summary.json', {'settings': summaries})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hac-file')
    parser.add_argument('--cas-file')
    parser.add_argument('--qtype', choices=['closed', 'open'], default='closed')
    parser.add_argument('--output-json', default=str(BENCH_ROOT / '04_results/json/comparison.json'))
    parser.add_argument('--output-md', default=str(BENCH_ROOT / '04_results/tables/comparison.md'))
    parser.add_argument('--hac-dir', default=str(BENCH_ROOT / '03_runs/hac/raw'))
    parser.add_argument('--cas-dir', default=str(BENCH_ROOT / '03_runs/cas_closed/features'))
    parser.add_argument('--output-dir', default=str(BENCH_ROOT / '04_results'))
    parser.add_argument('--outer-splits', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--splitter', choices=['kfold', 'stratified'], default='stratified')
    parser.add_argument('--bootstraps', type=int, default=1000)
    parser.add_argument('--include-cas-full', action='store_true', help=argparse.SUPPRESS)
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())
