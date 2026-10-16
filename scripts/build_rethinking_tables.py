#!/usr/bin/env python3
"""Build rethinking/shortcut evidence tables from cached benchmark rows.

This script does not run model inference. It uses closed-ended CAS/HAC caches to
summarize counterfactual answer behavior and high-confidence wrong enrichment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_common import BENCH_ROOT, finite, load_jsonl, normalize_yes_no, output_path, row_sample_id, write_json, write_jsonl  # noqa: E402

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
VARIANTS = ['real', 'textonly', 'black', 'noise', 'blur']
VARIANT_LABELS = {
    'real': 'Real image',
    'textonly': 'Text-only',
    'black': 'Black image',
    'noise': 'Noise image',
    'blur': 'Blur image',
}
TEMPLATE_LABELS = {
    '__all__': 'All closed',
    'generic_yesno': 'Generic yes/no',
    'anatomy': 'Anatomy',
    'abnormality_or_lesion': 'Abnormality/lesion',
    'contrast': 'Contrast',
    'view_or_plane': 'View/plane',
    'location': 'Location',
    'modality': 'Modality',
    'other_closed': 'Other closed',
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


def esc(value: Any) -> str:
    return str(value).replace('\\', r'\textbackslash{}').replace('_', r'\_').replace('&', r'\&').replace('%', r'\%')


def template_bucket(question: Any) -> str:
    q = str(question or '').strip().lower()
    if any(k in q for k in ('contrast', 'noncontrast', 'non-contrast')):
        return 'contrast'
    if any(k in q for k in ('ct scan', 'mri', 'x-ray', 'xray', 'ultrasound', 'modality', 'scan taken', 'radiograph')):
        return 'modality'
    if any(k in q for k in ('abnormal', 'disease', 'lesion', 'mass', 'tumor', 'edema', 'fracture', 'effusion', 'pneumonia', 'opacity', 'hemorrhage', 'infarct', 'nodule')):
        return 'abnormality_or_lesion'
    if any(k in q for k in ('left', 'right', 'upper', 'lower', 'location', 'where')):
        return 'location'
    if any(k in q for k in ('organ', 'kidney', 'liver', 'lung', 'heart', 'brain', 'spine', 'bone', 'spleen', 'bladder')):
        return 'anatomy'
    if any(k in q for k in ('view', 'plane', 'axial', 'coronal', 'sagittal')):
        return 'view_or_plane'
    if q.startswith(('is ', 'are ', 'does ', 'do ', 'can ', 'was ', 'were ', 'has ', 'have ')):
        return 'generic_yesno'
    return 'other_closed'


def pred_from_probs(probs: Mapping[str, Any]) -> str | None:
    if not finite(probs.get('p_yes')) or not finite(probs.get('p_no')):
        return None
    return 'yes' if float(probs.get('p_yes')) >= float(probs.get('p_no')) else 'no'


def max_conf(probs: Mapping[str, Any]) -> float | None:
    if not finite(probs.get('p_yes')) or not finite(probs.get('p_no')):
        return None
    return max(float(probs['p_yes']), float(probs['p_no']))


def variant_probs(row: Mapping[str, Any], variant: str) -> Mapping[str, Any]:
    if variant == 'real':
        return row.get('original') or {}
    return (row.get('baselines') or {}).get(variant) or {}


def iter_cas_rows(cas_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for dataset in DATASET_ORDER:
            path = output_path(cas_dir, model, dataset, 'closed')
            for row in load_jsonl(path):
                if row.get('status') == 'ok':
                    rows.append(row)
    return rows


def summarize_counterfactual_probe(rows: Sequence[Mapping[str, Any]], out_dir: Path, table_dir: Path) -> dict[str, Any]:
    per_variant: dict[str, list[dict[str, Any]]] = {v: [] for v in VARIANTS}
    per_setting: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    for row in rows:
        gt = normalize_yes_no(row.get('ground_truth'))
        real_pred = pred_from_probs(variant_probs(row, 'real'))
        for variant in VARIANTS:
            probs = variant_probs(row, variant)
            pred = pred_from_probs(probs)
            conf = max_conf(probs)
            record = {
                'model': row.get('model'),
                'dataset': row.get('dataset'),
                'sample_id': row_sample_id(row),
                'template': template_bucket(row.get('question')),
                'variant': variant,
                'prediction': pred,
                'ground_truth': gt,
                'is_correct': bool(pred == gt) if pred and gt else None,
                'confidence': conf,
                'yes': 1.0 if pred == 'yes' else 0.0 if pred == 'no' else None,
                'agreement_with_real': bool(pred == real_pred) if pred and real_pred else None,
            }
            raw_rows.append(record)
            per_variant[variant].append(record)
    summary_rows = []
    for variant in VARIANTS:
        vals = per_variant[variant]
        correct = [float(v['is_correct']) for v in vals if isinstance(v.get('is_correct'), bool)]
        conf = [float(v['confidence']) for v in vals if finite(v.get('confidence'))]
        yes = [float(v['yes']) for v in vals if finite(v.get('yes'))]
        agree = [float(v['agreement_with_real']) for v in vals if isinstance(v.get('agreement_with_real'), bool)]
        summary_rows.append({
            'variant': variant,
            'n': len(vals),
            'accuracy': float(np.mean(correct)) if correct else None,
            'mean_confidence': float(np.mean(conf)) if conf else None,
            'yes_rate': float(np.mean(yes)) if yes else None,
            'agreement_with_real': float(np.mean(agree)) if agree else None,
        })
    for model in MODEL_ORDER:
        for dataset in DATASET_ORDER:
            subset = [r for r in raw_rows if r.get('model') == model and r.get('dataset') == dataset]
            for variant in VARIANTS:
                vals = [r for r in subset if r.get('variant') == variant]
                if not vals:
                    continue
                correct = [float(v['is_correct']) for v in vals if isinstance(v.get('is_correct'), bool)]
                conf = [float(v['confidence']) for v in vals if finite(v.get('confidence'))]
                yes = [float(v['yes']) for v in vals if finite(v.get('yes'))]
                agree = [float(v['agreement_with_real']) for v in vals if isinstance(v.get('agreement_with_real'), bool)]
                per_setting.append({
                    'model': model,
                    'dataset': dataset,
                    'variant': variant,
                    'n': len(vals),
                    'accuracy': float(np.mean(correct)) if correct else None,
                    'mean_confidence': float(np.mean(conf)) if conf else None,
                    'yes_rate': float(np.mean(yes)) if yes else None,
                    'agreement_with_real': float(np.mean(agree)) if agree else None,
                })
    out = {'summary': summary_rows, 'per_setting': per_setting, 'raw_rows': raw_rows}
    dst = out_dir / 'counterfactual_probe'
    write_json(dst / 'summary.json', {'summary': summary_rows, 'per_setting': per_setting})
    write_jsonl(dst / 'rows.jsonl', raw_rows)
    write_csv(dst / 'per_setting.csv', per_setting)
    write_counterfactual_table(table_dir / 'shortcut_probe.tex', summary_rows)
    write_counterfactual_full_table(table_dir / 'shortcut_probe_full.tex', per_setting)
    return out


def load_hac_map(hac_dir: Path, model: str, dataset: str) -> dict[str, dict[str, Any]]:
    return {row_sample_id(r): r for r in load_jsonl(output_path(hac_dir, model, dataset, 'closed')) if r.get('status') == 'ok'}


def summarize_template_enrichment(rows: Sequence[Mapping[str, Any]], hac_dir: Path, out_dir: Path, table_dir: Path, *, hac_threshold: float, real_threshold: float, bootstraps: int, seed: int) -> dict[str, Any]:
    hac_maps = {(m, d): load_hac_map(hac_dir, m, d) for m in MODEL_ORDER for d in DATASET_ORDER}
    full_counts: Counter[str] = Counter()
    high_counts: Counter[str] = Counter()
    high_rows: list[dict[str, Any]] = []
    full_rows: list[dict[str, Any]] = []
    for row in rows:
        model = str(row.get('model'))
        dataset = str(row.get('dataset'))
        sid = row_sample_id(row)
        bucket = template_bucket(row.get('question'))
        full_counts[bucket] += 1
        full_counts['__all__'] += 1
        full_rows.append({'model': model, 'dataset': dataset, 'sample_id': sid, 'template': bucket})
        if row.get('is_correct') is not False:
            continue
        hac_row = hac_maps.get((model, dataset), {}).get(sid, {})
        hac_conf = float(hac_row.get('hac_confidence_n10')) if finite(hac_row.get('hac_confidence_n10')) else float('nan')
        target = normalize_yes_no(row.get('target_answer') or row.get('target_prediction'))
        original = row.get('original') or {}
        real_target_prob = float(original.get(f'p_{target}')) if target in {'yes', 'no'} and finite(original.get(f'p_{target}')) else float('nan')
        is_high = (math.isfinite(hac_conf) and hac_conf >= hac_threshold) or (math.isfinite(real_target_prob) and real_target_prob >= real_threshold)
        if not is_high:
            continue
        high_counts[bucket] += 1
        high_counts['__all__'] += 1
        high_rows.append({
            'model': model,
            'dataset': dataset,
            'sample_id': sid,
            'template': bucket,
            'question': row.get('question'),
            'ground_truth': row.get('ground_truth'),
            'target_answer': row.get('target_answer') or row.get('target_prediction'),
            'hac_confidence_n10': hac_conf,
            'real_target_prob': real_target_prob,
        })
    total_full = max(1, full_counts['__all__'])
    total_high = max(1, high_counts['__all__'])
    rng = np.random.default_rng(seed)
    buckets = sorted([b for b in full_counts if b != '__all__'], key=lambda b: (-full_counts[b], b))
    summary_rows = []
    high_templates = [r['template'] for r in high_rows]
    for bucket in buckets:
        full_share = full_counts[bucket] / total_full
        high_share = high_counts[bucket] / total_high
        enrichment = high_share / full_share if full_share > 0 else None
        ci_low = ci_high = None
        if high_templates and full_share > 0 and bootstraps > 0:
            vals = []
            arr = np.asarray(high_templates, dtype=object)
            for _ in range(bootstraps):
                sample = rng.choice(arr, size=len(arr), replace=True)
                vals.append(float(np.mean(sample == bucket)) / full_share)
            ci_low = float(np.percentile(vals, 2.5))
            ci_high = float(np.percentile(vals, 97.5))
        summary_rows.append({
            'template': bucket,
            'full_count': int(full_counts[bucket]),
            'full_share': float(full_share),
            'high_conf_wrong_count': int(high_counts[bucket]),
            'high_conf_wrong_share': float(high_share),
            'enrichment': float(enrichment) if enrichment is not None else None,
            'ci95_low': ci_low,
            'ci95_high': ci_high,
        })
    out = {
        'thresholds': {'hac_confidence_n10': hac_threshold, 'real_target_prob': real_threshold},
        'n_full': int(full_counts['__all__']),
        'n_high_conf_wrong': int(high_counts['__all__']),
        'summary': summary_rows,
    }
    dst = out_dir / 'template_enrichment'
    write_json(dst / 'summary.json', out)
    write_jsonl(dst / 'high_conf_wrong_rows.jsonl', high_rows)
    write_jsonl(dst / 'full_rows.jsonl', full_rows)
    write_csv(dst / 'summary.csv', summary_rows)
    write_enrichment_table(table_dir / 'template_enrichment.tex', out)
    write_enrichment_table(table_dir / 'high_conf_wrong.tex', out)
    return out


def build_template_prior_table(out_dir: Path, table_dir: Path) -> None:
    path = out_dir / 'template_prior' / 'summary.json'
    payload = json.loads(path.read_text(encoding='utf-8'))
    rows = payload.get('templates') or []
    rows = sorted(rows, key=lambda r: (0 if r.get('template') == '__all__' else 1, -int(r.get('n', 0))))
    lines = ["""\\begin{table}[t]
\\centering
\\caption{Closed-question template priors. Majority rate measures the best constant yes/no answer within a template bucket.}
\\label{tab:template_prior}
\\small
\\begin{tabular}{lrrr}
\\toprule
Template & $n$ & Yes rate & Majority rate \\\\
\\midrule
"""]
    for row in rows:
        template = TEMPLATE_LABELS.get(row.get('template'), str(row.get('template')))
        lines.append(f"{esc(template)} & {int(row.get('n', 0))} & {fmt(row.get('yes_rate'))} & {fmt(row.get('majority_rate'))} " + r"\\" + "\n")
    lines.append("""\\bottomrule
\\end{tabular}
\\end{table}
""")
    (table_dir / 'template_prior.tex').write_text(''.join(lines), encoding='utf-8')


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_counterfactual_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = ["""\\begin{table}[t]
\\centering
\\caption{Full closed-ended counterfactual shortcut probe from cached CAS rows across 21 model-dataset settings. Agreement is with the real-image binary prediction.}
\\label{tab:shortcut_probe}
\\small
\\begin{tabular}{lrrrr}
\\toprule
Variant & $n$ & Accuracy & Mean confidence & Yes rate \\\\
\\midrule
"""]
    for row in rows:
        lines.append(
            f"{esc(VARIANT_LABELS.get(row['variant'], row['variant']))} & {row['n']} & {fmt(row.get('accuracy'))} & "
            f"{fmt(row.get('mean_confidence'))} & {fmt(row.get('yes_rate'))} " + r"\\" + "\n"
        )
    lines.append("""\\bottomrule
\\end{tabular}
\\end{table}
""")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(lines), encoding='utf-8')


def write_counterfactual_full_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = ["""\\begin{longtable}{lllrrrr}
\\caption{Per-setting closed-ended counterfactual shortcut probe from cached CAS rows.}\\label{tab:shortcut_probe_full}\\
\\toprule
Model & Dataset & Variant & $n$ & Accuracy & Confidence & Yes rate \\\\
\\midrule
\\endfirsthead
\\toprule
Model & Dataset & Variant & $n$ & Accuracy & Confidence & Yes rate \\\\
\\midrule
\\endhead
"""]
    for row in rows:
        lines.append(
            f"{esc(MODEL_LABELS.get(row['model'], row['model']))} & {esc(DATASET_LABELS.get(row['dataset'], row['dataset']))} & "
            f"{esc(VARIANT_LABELS.get(row['variant'], row['variant']))} & {row['n']} & {fmt(row.get('accuracy'))} & "
            f"{fmt(row.get('mean_confidence'))} & {fmt(row.get('yes_rate'))} " + r"\\" + "\n"
        )
    lines.append("""\\bottomrule
\\end{longtable}
""")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(lines), encoding='utf-8')


def weighted_mean(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    num = 0.0
    den = 0.0
    for row in rows:
        value = row.get(field)
        weight = row.get('n', 0)
        if not finite(value):
            continue
        try:
            w = float(weight)
        except (TypeError, ValueError):
            continue
        if w <= 0:
            continue
        num += float(value) * w
        den += w
    return num / den if den > 0 else None


def summarize_binary_shortcut_summaries(out_dir: Path, table_dir: Path) -> dict[str, Any] | None:
    src = out_dir / 'binary_shortcut' / 'summary'
    if not src.exists():
        return None
    per_setting: list[dict[str, Any]] = []
    for path in sorted(src.glob('*.json')):
        parts = path.stem.split('__')
        if len(parts) != 3:
            continue
        model, dataset, qtype = parts
        if qtype != 'closed' or model not in MODEL_LABELS or dataset not in DATASET_LABELS:
            continue
        payload = json.loads(path.read_text(encoding='utf-8'))
        variants = payload.get('variants') or {}
        agreement = payload.get('agreement') or {}
        for variant in ('real', 'textonly', 'black', 'random_image'):
            row = variants.get(variant)
            if not isinstance(row, Mapping):
                continue
            per_setting.append({
                'model': model,
                'dataset': dataset,
                'variant': variant,
                'n': int(row.get('n') or 0),
                'accuracy': row.get('accuracy'),
                'mean_confidence': row.get('mean_binary_confidence'),
                'yes_rate': row.get('yes_rate'),
                'agreement_with_real': 1.0 if variant == 'real' else agreement.get(f'real_vs_{variant}'),
                'source': str(path),
            })
    if not per_setting:
        return None
    summary = []
    for variant in ('real', 'textonly', 'black', 'random_image'):
        rows = [r for r in per_setting if r.get('variant') == variant]
        if not rows:
            continue
        summary.append({
            'variant': variant,
            'settings': len(rows),
            'n': int(sum(int(r.get('n') or 0) for r in rows)),
            'accuracy': weighted_mean(rows, 'accuracy'),
            'mean_confidence': weighted_mean(rows, 'mean_confidence'),
            'yes_rate': weighted_mean(rows, 'yes_rate'),
            'agreement_with_real': weighted_mean(rows, 'agreement_with_real'),
        })
    payload = {'summary': summary, 'per_setting': per_setting}
    dst = out_dir / 'binary_shortcut'
    write_json(dst / 'aggregate_summary.json', payload)
    write_csv(dst / 'aggregate_per_setting.csv', per_setting)
    write_binary_shortcut_table(table_dir / 'shortcut_probe_random.tex', summary)
    write_binary_shortcut_full_table(table_dir / 'shortcut_probe_random_full.tex', per_setting)
    return payload


def binary_variant_label(variant: Any) -> str:
    labels = {
        'real': 'Real image',
        'textonly': 'Text-only',
        'black': 'Black image',
        'random_image': 'Random image',
    }
    return labels.get(str(variant), str(variant))


def write_binary_shortcut_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = ["""\\begin{table}[t]
\\centering
\\caption{Direct closed-ended binary shortcut probe. Random image replaces the paired image with another dataset image while keeping the same question. Results aggregate completed model-dataset settings only.}
\\label{tab:shortcut_probe_random}
\\small
\\begin{tabular}{lrrrrr}
\\toprule
Variant & Settings & $n$ & Accuracy & Confidence & Yes rate \\\\
\\midrule
"""]
    for row in rows:
        lines.append(
            f"{esc(binary_variant_label(row.get('variant')))} & {int(row.get('settings', 0))} & {int(row.get('n', 0))} & "
            f"{fmt(row.get('accuracy'))} & {fmt(row.get('mean_confidence'))} & {fmt(row.get('yes_rate'))} " + r"\\" + "\n"
        )
    lines.append("""\\bottomrule
\\end{tabular}
\\end{table}
""")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(lines), encoding='utf-8')


def write_binary_shortcut_full_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = ["""\\begin{longtable}{lllrrrr}
\\caption{Per-setting direct binary shortcut probe with random-image replacement.}\\label{tab:shortcut_probe_random_full}\\
\\toprule
Model & Dataset & Variant & $n$ & Accuracy & Confidence & Yes rate \\\\
\\midrule
\\endfirsthead
\\toprule
Model & Dataset & Variant & $n$ & Accuracy & Confidence & Yes rate \\\\
\\midrule
\\endhead
"""]
    order = {v: i for i, v in enumerate(('real', 'textonly', 'black', 'random_image'))}
    sorted_rows = sorted(rows, key=lambda r: (
        MODEL_ORDER.index(r['model']) if r.get('model') in MODEL_ORDER else 99,
        DATASET_ORDER.index(r['dataset']) if r.get('dataset') in DATASET_ORDER else 99,
        order.get(str(r.get('variant')), 99),
    ))
    for row in sorted_rows:
        lines.append(
            f"{esc(MODEL_LABELS.get(row['model'], row['model']))} & {esc(DATASET_LABELS.get(row['dataset'], row['dataset']))} & "
            f"{esc(binary_variant_label(row.get('variant')))} & {int(row.get('n', 0))} & {fmt(row.get('accuracy'))} & "
            f"{fmt(row.get('mean_confidence'))} & {fmt(row.get('yes_rate'))} " + r"\\" + "\n"
        )
    lines.append("""\\bottomrule
\\end{longtable}
""")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(lines), encoding='utf-8')


def write_enrichment_table(path: Path, payload: Mapping[str, Any]) -> None:
    lines = ["""\\begin{table}[t]
\\centering
\\caption{Template enrichment among high-confidence wrong closed-ended answers. High-confidence wrong means an incorrect greedy answer with HAC confidence $\\geq 0.9$ or real-image target probability $\\geq 0.9$.}
\\label{tab:template_enrichment}
\\small
\\begin{tabular}{lrrrr}
\\toprule
Template & Full share & HC-wrong share & Enrichment & CI95 \\\\
\\midrule
"""]
    rows = sorted(payload.get('summary') or [], key=lambda r: (-(r.get('enrichment') or -1), -int(r.get('high_conf_wrong_count', 0))))
    for row in rows:
        if int(row.get('high_conf_wrong_count', 0)) == 0:
            continue
        ci = '--'
        if row.get('ci95_low') is not None and row.get('ci95_high') is not None:
            ci = f"[{fmt(row.get('ci95_low'))}, {fmt(row.get('ci95_high'))}]"
        template = TEMPLATE_LABELS.get(row.get('template'), str(row.get('template')))
        lines.append(
            f"{esc(template)} & {fmt(row.get('full_share'))} & {fmt(row.get('high_conf_wrong_share'))} & "
            f"{fmt(row.get('enrichment'))} & {ci} " + r"\\" + "\n"
        )
    lines.append("""\\bottomrule
\\end{tabular}
\\end{table}
""")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(lines), encoding='utf-8')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cas-closed-dir', default=str(BENCH_ROOT / '03_runs/cas_closed/features'))
    parser.add_argument('--hac-dir', default=str(BENCH_ROOT / '03_runs/hac/raw'))
    parser.add_argument('--output-dir', default=str(BENCH_ROOT / '06_shortcut_validation'))
    parser.add_argument('--table-dir', default=str(BENCH_ROOT / '08_overleaf_draft_local/tables'))
    parser.add_argument('--hac-conf-threshold', type=float, default=0.9)
    parser.add_argument('--real-conf-threshold', type=float, default=0.9)
    parser.add_argument('--bootstraps', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    table_dir = Path(args.table_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    rows = iter_cas_rows(Path(args.cas_closed_dir))
    print(f'loaded closed CAS rows: {len(rows)}')
    build_template_prior_table(out_dir, table_dir)
    direct_binary = summarize_binary_shortcut_summaries(out_dir, table_dir)
    cf = summarize_counterfactual_probe(rows, out_dir, table_dir)
    enrich = summarize_template_enrichment(
        rows,
        Path(args.hac_dir),
        out_dir,
        table_dir,
        hac_threshold=args.hac_conf_threshold,
        real_threshold=args.real_conf_threshold,
        bootstraps=args.bootstraps,
        seed=args.seed,
    )
    if direct_binary is not None:
        print(f"direct binary settings: {len(direct_binary['per_setting'])}")
    print(f"counterfactual variants: {len(cf['summary'])}")
    print(f"high-confidence wrong rows: {enrich['n_high_conf_wrong']} / {enrich['n_full']}")
    print(f'wrote rethinking tables to {table_dir}')


if __name__ == '__main__':
    main()
