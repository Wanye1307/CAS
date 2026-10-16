#!/usr/bin/env python3
"""Feature-level ablations for closed-ended CAS.

This script does not run model inference. It evaluates cached CAS closed feature
JSONL files on the same HAC/CAS sample intersection used by the paper tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_common import BENCH_ROOT, load_jsonl, output_path, write_json  # noqa: E402
from build_paper_metric_tables import (  # noqa: E402
    DATASET_LABELS,
    DATASET_ORDER,
    MODEL_LABELS,
    MODEL_ORDER,
    auc,
    build_intersection,
    logreg_cv,
    metric_bundle,
    row_features,
)

CAS_FEATURES_17 = [
    'cas_pred_text', 'cas_pred_black', 'cas_pred_noise', 'cas_pred_blur',
    'cas_opp_text', 'cas_opp_black', 'cas_opp_noise', 'cas_opp_blur',
    'cas_pred_mean', 'cas_pred_max', 'cas_pred_min',
    'cas_opp_mean', 'cas_opp_max', 'cas_opp_min',
    'cas_cf_support_mean', 'cas_cf_support_max', 'cas_same_pred_frac',
]

# This is the feature set used by the current closed CAS main result script.
CAS_PAPER_9 = [
    'cas_pred_mean', 'cas_pred_max', 'cas_pred_min',
    'cas_opp_mean', 'cas_opp_max', 'cas_opp_min',
    'cas_cf_support_mean', 'cas_cf_support_max', 'cas_same_pred_frac',
]

GROUPS = {
    'CAS-paper-9D': CAS_PAPER_9,
    'CAS-full-17D': CAS_FEATURES_17,
    'Pred shift only': [
        'cas_pred_text', 'cas_pred_black', 'cas_pred_noise', 'cas_pred_blur',
        'cas_pred_mean', 'cas_pred_max', 'cas_pred_min',
    ],
    'Opp shift only': [
        'cas_opp_text', 'cas_opp_black', 'cas_opp_noise', 'cas_opp_blur',
        'cas_opp_mean', 'cas_opp_max', 'cas_opp_min',
    ],
    'View raw shifts': [
        'cas_pred_text', 'cas_pred_black', 'cas_pred_noise', 'cas_pred_blur',
        'cas_opp_text', 'cas_opp_black', 'cas_opp_noise', 'cas_opp_blur',
    ],
    'Aggregate shifts': [
        'cas_pred_mean', 'cas_pred_max', 'cas_pred_min',
        'cas_opp_mean', 'cas_opp_max', 'cas_opp_min',
    ],
    'CF support': ['cas_cf_support_mean', 'cas_cf_support_max'],
    'Same-pred stability': ['cas_same_pred_frac'],
    'Text view': ['cas_pred_text', 'cas_opp_text'],
    'Black view': ['cas_pred_black', 'cas_opp_black'],
    'Noise view': ['cas_pred_noise', 'cas_opp_noise'],
    'Blur view': ['cas_pred_blur', 'cas_opp_blur'],
}

PRIMARY_MODELS = [m for m in MODEL_ORDER if 'llava' not in m.lower()]


def closed_setting_files(hac_dir: Path, cas_dir: Path) -> list[tuple[str, str, Path, Path]]:
    out = []
    for cas_file in sorted(cas_dir.glob('*__closed.jsonl')):
        stem = cas_file.name[:-len('__closed.jsonl')]
        if '__' not in stem:
            continue
        model, dataset = stem.split('__', 1)
        hac_file = output_path(hac_dir, model, dataset, 'closed')
        if not hac_file.exists():
            continue
        out.append((model, dataset, hac_file, cas_file))
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    ds_order = {d: i for i, d in enumerate(DATASET_ORDER)}
    out.sort(key=lambda x: (order.get(x[0], 99), ds_order.get(x[1], 99), x[0], x[1]))
    return out


def finite_float(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float('nan')
    return x if math.isfinite(x) else float('nan')


def fmt(value: Any, digits: int = 3) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return '--'
    if not math.isfinite(x):
        return '--'
    return f'{x:.{digits}f}'


def esc(s: Any) -> str:
    return str(s).replace('\\', r'\textbackslash{}').replace('_', r'\_').replace('&', r'\&').replace('%', r'\%')


def feature_vector(rows: Sequence[Mapping[str, Any]], feature: str) -> np.ndarray:
    vals = []
    for row in rows:
        vals.append(finite_float((row.get('features') or {}).get(feature)))
    return np.asarray(vals, dtype=float)


def make_metric_row(model: str, dataset: str, family: str, method: str, labels: np.ndarray, risk: Sequence[float], *, features: Sequence[str] | None = None, orientation: str = 'fixed') -> dict[str, Any]:
    metrics = metric_bundle(labels, risk)
    row = {
        'model': model,
        'dataset': dataset,
        'family': family,
        'method': method,
        'orientation': orientation,
        'features': ';'.join(features or []),
        'n': int(len(labels)),
    }
    row.update(metrics)
    return row


def single_feature_rows(model: str, dataset: str, labels: np.ndarray, cas_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for feat in CAS_FEATURES_17:
        scores = feature_vector(cas_rows, feat)
        fixed = make_metric_row(model, dataset, 'single_feature', feat, labels, scores, features=[feat], orientation='fixed')
        fixed_auc = fixed.get('auroc')
        flipped_scores = -scores
        flipped = make_metric_row(model, dataset, 'single_feature', feat, labels, flipped_scores, features=[feat], orientation='flipped')
        flipped_auc = flipped.get('auroc')
        fixed['auroc_flipped'] = flipped_auc
        if fixed_auc is None or not math.isfinite(float(fixed_auc)):
            fixed['auroc_oracle'] = flipped_auc
            fixed['oracle_direction'] = 'flipped'
        elif flipped_auc is None or not math.isfinite(float(flipped_auc)):
            fixed['auroc_oracle'] = fixed_auc
            fixed['oracle_direction'] = 'fixed'
        elif float(flipped_auc) > float(fixed_auc):
            fixed['auroc_oracle'] = flipped_auc
            fixed['oracle_direction'] = 'flipped'
        else:
            fixed['auroc_oracle'] = fixed_auc
            fixed['oracle_direction'] = 'fixed'
        out.append(fixed)
    return out


def group_rows(model: str, dataset: str, labels: np.ndarray, cas_rows: list[dict[str, Any]], n_splits: int, seed: int, splitter: str) -> list[dict[str, Any]]:
    out = []
    for name, feats in GROUPS.items():
        rep = logreg_cv(labels, row_features(cas_rows, feats), n_splits, seed, splitter)
        out.append(make_metric_row(model, dataset, 'group_lr', name, labels, rep.get('scores_oof'), features=feats, orientation='cv_lr'))
    return out


def leave_one_out_rows(model: str, dataset: str, labels: np.ndarray, cas_rows: list[dict[str, Any]], n_splits: int, seed: int, splitter: str) -> list[dict[str, Any]]:
    out = []
    for feat in CAS_FEATURES_17:
        feats = [f for f in CAS_FEATURES_17 if f != feat]
        rep = logreg_cv(labels, row_features(cas_rows, feats), n_splits, seed, splitter)
        out.append(make_metric_row(model, dataset, 'leave_one_out_lr', f'w/o {feat}', labels, rep.get('scores_oof'), features=feats, orientation='cv_lr'))
    return out


def aggregate(rows: list[dict[str, Any]], *, include_llava: bool = False, group_keys: Sequence[str]) -> list[dict[str, Any]]:
    use = [r for r in rows if include_llava or r['model'] in PRIMARY_MODELS]
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in use:
        buckets[tuple(r.get(k) for k in group_keys)].append(r)
    out = []
    metric_keys = ['auroc', 'auroc_flipped', 'auroc_oracle', 'aurc', 'eaurc', 'accuracy', 'base_risk']
    for key, vals in sorted(buckets.items()):
        rec = {k: v for k, v in zip(group_keys, key)}
        rec['settings'] = len(vals)
        rec['include_llava'] = include_llava
        for mk in metric_keys:
            xs = [finite_float(v.get(mk)) for v in vals]
            xs = [x for x in xs if math.isfinite(x)]
            rec[f'mean_{mk}'] = float(np.mean(xs)) if xs else None
            rec[f'std_{mk}'] = float(np.std(xs)) if xs else None
        rec['wins_fixed_over_0.5'] = sum(1 for v in vals if math.isfinite(finite_float(v.get('auroc'))) and finite_float(v.get('auroc')) > 0.5)
        rec['wins_oracle_over_0.5'] = sum(1 for v in vals if math.isfinite(finite_float(v.get('auroc_oracle'))) and finite_float(v.get('auroc_oracle')) > 0.5)
        out.append(rec)
    return out


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('')
        return
    fields = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_single_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    # rows are aggregated by method and dataset over the primary 7 models.
    by_feat = {r['method']: r for r in aggregate(rows, group_keys=['method']) if r.get('method')}
    by_ds = {(r['method'], r['dataset']): r for r in aggregate(rows, group_keys=['method', 'dataset'])}
    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{Single-feature ablation of closed-ended CAS. AUROC is computed for error detection using each feature alone on the same HAC/CAS sample intersection as the main benchmark. Fixed uses the raw feature direction; oracle reports $\max(\mathrm{AUC},1-\mathrm{AUC})$ for diagnostic strength only.}')
    lines.append(r'\label{tab:cas_single_feature_ablation}')
    lines.append(r'\small')
    lines.append(r'\begin{tabular}{lccccc}')
    lines.append(r'\toprule')
    lines.append(r'Feature & VQA-RAD & SLAKE & VQA-Med2019 & Fixed macro & Oracle macro \\')
    lines.append(r'\midrule')
    for feat in CAS_FEATURES_17:
        vals = []
        for ds in DATASET_ORDER:
            vals.append(fmt((by_ds.get((feat, ds)) or {}).get('mean_auroc')))
        macro = by_feat.get(feat, {})
        lines.append(f'{esc(feat)} & {vals[0]} & {vals[1]} & {vals[2]} & {fmt(macro.get("mean_auroc"))} & {fmt(macro.get("mean_auroc_oracle"))} \\\\')
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    path.write_text('\n'.join(lines) + '\n')


def write_group_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    agg = aggregate(rows, group_keys=['method'])
    by_method = {r['method']: r for r in agg}
    order = list(GROUPS.keys())
    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{Group-level CAS ablation with the same 5-fold cross-validated logistic-regression protocol used in the main closed-ended benchmark. Metrics are macro-averaged over 7 models and 3 datasets.}')
    lines.append(r'\label{tab:cas_group_ablation}')
    lines.append(r'\small')
    lines.append(r'\begin{tabular}{lccc}')
    lines.append(r'\toprule')
    lines.append(r'Feature set & AUROC $\uparrow$ & AURC $\downarrow$ & E-AURC $\downarrow$ \\')
    lines.append(r'\midrule')
    for name in order:
        r = by_method.get(name, {})
        lines.append(f'{esc(name)} & {fmt(r.get("mean_auroc"))} & {fmt(r.get("mean_aurc"))} & {fmt(r.get("mean_eaurc"))} \\\\')
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    path.write_text('\n'.join(lines) + '\n')


def write_leave_one_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    full_rows = [r for r in rows if r['family'] == 'group_lr' and r['method'] == 'CAS-full-17D']
    loo_rows = [r for r in rows if r['family'] == 'leave_one_out_lr']
    full_macro = aggregate(full_rows, group_keys=['method'])[0]['mean_auroc'] if full_rows else None
    agg = aggregate(loo_rows, group_keys=['method'])
    for r in agg:
        r['drop_vs_full'] = None if full_macro is None or r.get('mean_auroc') is None else float(full_macro - r['mean_auroc'])
    agg = sorted(agg, key=lambda r: finite_float(r.get('drop_vs_full')), reverse=True)
    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{Leave-one-feature-out CAS ablation. Positive $\Delta$ means removing the feature lowers AUROC relative to CAS-full-17D.}')
    lines.append(r'\label{tab:cas_leave_one_feature_out}')
    lines.append(r'\small')
    lines.append(r'\begin{tabular}{lcc}')
    lines.append(r'\toprule')
    lines.append(r'Removed feature & AUROC & $\Delta$ vs full \\')
    lines.append(r'\midrule')
    for r in agg:
        lines.append(f'{esc(str(r["method"]).replace("w/o ", ""))} & {fmt(r.get("mean_auroc"))} & {fmt(r.get("drop_vs_full"))} \\\\')
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    path.write_text('\n'.join(lines) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--hac-dir', type=Path, default=BENCH_ROOT / '03_runs/hac/raw')
    parser.add_argument('--cas-dir', type=Path, default=BENCH_ROOT / '03_runs/cas_closed/features')
    parser.add_argument('--output-dir', type=Path, default=BENCH_ROOT / '04_results/cas_feature_ablation')
    parser.add_argument('--outer-splits', type=int, default=5)
    parser.add_argument('--splitter', choices=['stratified', 'kfold'], default='stratified')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_single = []
    all_group = []
    all_loo = []
    manifest = []

    for model, dataset, hac_file, cas_file in closed_setting_files(args.hac_dir, args.cas_dir):
        hac_rows = load_jsonl(hac_file)
        cas_rows = load_jsonl(cas_file)
        hac_rows, cas_rows, ids = build_intersection(hac_rows, cas_rows)
        labels = np.asarray([0 if bool(r.get('is_correct')) else 1 for r in cas_rows], dtype=int)
        manifest.append({
            'model': model,
            'dataset': dataset,
            'n': int(len(labels)),
            'n_error': int(np.sum(labels == 1)),
            'accuracy': float(np.mean(labels == 0)) if len(labels) else None,
            'hac_file': str(hac_file),
            'cas_file': str(cas_file),
        })
        all_single.extend(single_feature_rows(model, dataset, labels, cas_rows))
        all_group.extend(group_rows(model, dataset, labels, cas_rows, args.outer_splits, args.seed, args.splitter))
        all_loo.extend(leave_one_out_rows(model, dataset, labels, cas_rows, args.outer_splits, args.seed, args.splitter))

    write_csv(args.output_dir / 'single_feature_17_per_setting.csv', all_single)
    write_csv(args.output_dir / 'group_lr_per_setting.csv', all_group)
    write_csv(args.output_dir / 'leave_one_feature_out_per_setting.csv', all_loo)
    write_csv(args.output_dir / 'manifest.csv', manifest)

    single_summary = aggregate(all_single, group_keys=['method'])
    single_dataset_summary = aggregate(all_single, group_keys=['method', 'dataset'])
    group_summary = aggregate(all_group, group_keys=['method'])
    loo_summary = aggregate(all_loo, group_keys=['method'])
    all_single_summary = aggregate(all_single, include_llava=True, group_keys=['method'])
    all_group_summary = aggregate(all_group, include_llava=True, group_keys=['method'])

    write_csv(args.output_dir / 'single_feature_17_macro_summary.csv', single_summary)
    write_csv(args.output_dir / 'single_feature_17_by_dataset_summary.csv', single_dataset_summary)
    write_csv(args.output_dir / 'group_lr_macro_summary.csv', group_summary)
    write_csv(args.output_dir / 'leave_one_feature_out_macro_summary.csv', loo_summary)
    write_csv(args.output_dir / 'single_feature_17_macro_summary_with_llava.csv', all_single_summary)
    write_csv(args.output_dir / 'group_lr_macro_summary_with_llava.csv', all_group_summary)

    write_single_latex(args.output_dir / 'table_single_feature_17.tex', all_single)
    write_group_latex(args.output_dir / 'table_group_ablation.tex', all_group)
    write_leave_one_latex(args.output_dir / 'table_leave_one_feature_out.tex', all_group + all_loo)

    write_json(args.output_dir / 'cas_feature_ablation_manifest.json', {
        'protocol': 'closed CAS feature ablation from cached greedy HAC/CAS intersection',
        'primary_models': PRIMARY_MODELS,
        'datasets': DATASET_ORDER,
        'features_17': CAS_FEATURES_17,
        'paper_main_features_9': CAS_PAPER_9,
        'groups': GROUPS,
        'outer_splits': args.outer_splits,
        'splitter': args.splitter,
        'seed': args.seed,
        'outputs': sorted(p.name for p in args.output_dir.iterdir()),
    })

    print(f'Wrote CAS feature ablation artifacts to {args.output_dir}')
    print('Primary non-LLaVA settings:', sum(1 for m in manifest if m['model'] in PRIMARY_MODELS))
    print('All settings:', len(manifest))


if __name__ == '__main__':
    main()
