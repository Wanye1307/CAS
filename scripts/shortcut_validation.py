#!/usr/bin/env python3
"""Shortcut validation probes for closed-ended medical VQA.

The probes test whether yes/no questions can be answered from textual/template
priors rather than the paired medical image.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_common import (  # noqa: E402
    BENCH_ROOT,
    append_jsonl,
    done_indices,
    judge_correct,
    load_dataset,
    load_jsonl,
    normalize_yes_no,
    output_path,
    parse_generation,
    row_sample_id,
    sample_id_for,
    sanitize,
    set_seed,
    write_json,
    write_jsonl,
)

import medvlm  # noqa: E402
from confidence.grounding import CLOSED_GROUNDING_PROMPT, _binary_next_token_probs, make_baseline_image  # noqa: E402


DEFAULT_OUT = BENCH_ROOT / '06_shortcut_validation'
DATASETS = ('vqa_rad', 'slake', 'vqa_med_2019')


def fmt(value: Any) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 'NA'
    return f'{f:.4f}' if math.isfinite(f) else 'NA'


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


def load_label_map(label_dir: Path, model: str, dataset: str) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(output_path(label_dir, model, dataset, 'closed'))
    return {row_sample_id(r): r for r in rows}


def template_prior(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {'datasets': {}, 'templates': {}}
    global_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for dataset in args.datasets:
        rows = []
        ds = load_dataset(dataset, 'closed', args.seed, args.max_samples)
        for idx, sample in enumerate(ds):
            gt = normalize_yes_no(sample.get('answer'))
            if gt not in {'yes', 'no'}:
                continue
            bucket = template_bucket(sample.get('question'))
            rows.append({'dataset': dataset, 'dataset_index': idx, 'sample_id': sample_id_for(idx, sample), 'template': bucket, 'ground_truth': gt, 'question': sample.get('question')})
            global_counts[bucket][gt] += 1
            global_counts[bucket]['total'] += 1
            global_counts['__all__'][gt] += 1
            global_counts['__all__']['total'] += 1
        out['datasets'][dataset] = {'n': len(rows), 'rows': rows}
    template_rows = []
    for bucket, c in sorted(global_counts.items()):
        total = int(c['total'])
        if total <= 0:
            continue
        yes = int(c['yes'])
        no = int(c['no'])
        majority = max(yes, no) / total
        template_rows.append({'template': bucket, 'n': total, 'yes': yes, 'no': no, 'yes_rate': yes / total, 'majority_rate': majority})
    out['templates'] = template_rows
    out_dir = Path(args.output_dir) / 'template_prior'
    write_json(out_dir / 'summary.json', out)
    write_jsonl(out_dir / 'rows.jsonl', [r for d in out['datasets'].values() for r in d['rows']])
    write_template_markdown(out_dir / 'summary.md', template_rows)
    return out


def write_template_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        '# Closed Question Template Priors',
        '',
        '| template | n | yes | no | yes rate | majority rate |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for r in sorted(rows, key=lambda x: (-x['n'], x['template'])):
        lines.append(f"| {r['template']} | {r['n']} | {r['yes']} | {r['no']} | {fmt(r['yes_rate'])} | {fmt(r['majority_rate'])} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def high_conf_wrong(args: argparse.Namespace) -> dict[str, Any]:
    hac_dir = Path(args.hac_dir)
    label_dir = Path(args.label_dir)
    greedy_dir = Path(args.greedy_dir)
    cas_dir = Path(args.cas_closed_dir)
    rows = []
    template_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for model in args.models:
        for dataset in args.datasets:
            labels = load_label_map(label_dir, model, dataset)
            greedy_by_id = {row_sample_id(r): r for r in load_jsonl(output_path(greedy_dir, model, dataset, 'closed')) if r.get('status') == 'ok'}
            cas_by_id = {row_sample_id(r): r for r in load_jsonl(output_path(cas_dir, model, dataset, 'closed')) if r.get('status') == 'ok'}
            for h in load_jsonl(output_path(hac_dir, model, dataset, 'closed')):
                if h.get('status') != 'ok':
                    continue
                sid = row_sample_id(h)
                lab = labels.get(sid)
                is_correct = lab.get('is_correct') if lab else h.get('is_correct')
                if is_correct is not False:
                    continue
                hac_conf = h.get('hac_confidence_n10')
                try:
                    hac_conf_f = float(hac_conf)
                except (TypeError, ValueError):
                    hac_conf_f = float('nan')
                cas_row = cas_by_id.get(sid) or {}
                original = cas_row.get('original') or {}
                pred = normalize_yes_no((greedy_by_id.get(sid) or h).get('target_prediction') or (greedy_by_id.get(sid) or h).get('target_answer'))
                real_target_prob = None
                if pred == 'yes' and 'p_yes' in original:
                    real_target_prob = float(original['p_yes'])
                if pred == 'no' and 'p_no' in original:
                    real_target_prob = float(original['p_no'])
                if not (hac_conf_f >= args.hac_conf_threshold or (real_target_prob is not None and real_target_prob >= args.real_conf_threshold)):
                    continue
                bucket = template_bucket(h.get('question') or (greedy_by_id.get(sid) or {}).get('question'))
                row = {
                    'model': model,
                    'dataset': dataset,
                    'sample_id': sid,
                    'dataset_index': h.get('dataset_index'),
                    'template': bucket,
                    'question': h.get('question'),
                    'ground_truth': h.get('ground_truth'),
                    'prediction': (greedy_by_id.get(sid) or h).get('target_prediction') or (greedy_by_id.get(sid) or h).get('target_answer'),
                    'hac_confidence_n10': hac_conf_f,
                    'real_target_prob': real_target_prob,
                }
                rows.append(row)
                template_counts[bucket]['high_conf_wrong'] += 1
                template_counts[bucket]['total'] += 1
                template_counts['__all__']['high_conf_wrong'] += 1
                template_counts['__all__']['total'] += 1
    summary_rows = [{'template': k, **dict(v)} for k, v in sorted(template_counts.items())]
    out = {'rows': rows, 'summary': summary_rows, 'thresholds': {'hac_confidence_n10': args.hac_conf_threshold, 'real_target_prob': args.real_conf_threshold}}
    out_dir = Path(args.output_dir) / 'high_conf_wrong'
    write_json(out_dir / 'summary.json', out)
    write_jsonl(out_dir / 'rows.jsonl', rows)
    write_high_conf_markdown(out_dir / 'summary.md', rows, summary_rows)
    return out


def write_high_conf_markdown(path: Path, rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        '# High-Confidence Wrong Template Enrichment',
        '',
        f'- high-confidence wrong rows: {len(rows)}',
        '',
        '| template | count | share |',
        '|---|---:|---:|',
    ]
    total = max(1, len(rows))
    for r in sorted(summary_rows, key=lambda x: -int(x.get('high_conf_wrong', 0))):
        if r['template'] == '__all__':
            continue
        count = int(r.get('high_conf_wrong', 0))
        lines.append(f"| {r['template']} | {count} | {fmt(count / total)} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def pred_from_probs(probs: Mapping[str, Any]) -> str:
    return 'yes' if float(probs.get('p_yes', 0.0)) >= float(probs.get('p_no', 0.0)) else 'no'


def binary_shortcut(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir) / 'binary_shortcut'
    out_dir.mkdir(parents=True, exist_ok=True)
    for model_name in args.models:
        print(f'Loading model {model_name}', flush=True)
        model, processor, model_config = medvlm.load_model(
            model_name,
            quantization=args.quantization,
            use_flash_attention=not args.no_flash_attention,
            device_map=args.device_map,
        )
        for dataset in args.datasets:
            ds = load_dataset(dataset, 'closed', args.seed, args.max_samples)
            out_path = output_path(out_dir / 'raw', model_name, dataset, 'closed')
            done = done_indices(out_path) if args.resume else set()
            rng = np.random.default_rng(args.seed)
            perm = rng.permutation(len(ds))
            if len(ds) > 1:
                for i in range(len(perm)):
                    if int(perm[i]) == i:
                        perm[i] = int((i + 1) % len(ds))
            for idx, sample in enumerate(ds):
                if idx in done:
                    continue
                set_seed(args.seed * 1_000_003 + idx)
                try:
                    question = str(sample['question'])
                    gt = normalize_yes_no(sample.get('answer'))
                    prompt = CLOSED_GROUNDING_PROMPT.format(question=question)
                    real = _binary_next_token_probs(model, processor, model_config, sample['image'], prompt)
                    textonly = _binary_next_token_probs(model, processor, model_config, None, prompt)
                    black_image = make_baseline_image(sample['image'], baseline='black')
                    black = _binary_next_token_probs(model, processor, model_config, black_image, prompt)
                    rand_idx = int(perm[idx])
                    random_image = ds[rand_idx]['image']
                    random_probs = _binary_next_token_probs(model, processor, model_config, random_image, prompt)
                    variants = {'real': real, 'textonly': textonly, 'black': black, 'random_image': random_probs}
                    row = {
                        'version': 'shortcut_binary_probs_v1',
                        'status': 'ok',
                        'model': model_name,
                        'dataset': dataset,
                        'qtype': 'closed',
                        'dataset_index': idx,
                        'sample_id': sample_id_for(idx, sample),
                        'random_dataset_index': rand_idx,
                        'question_id': sample.get('question_id'),
                        'image_id': sample.get('image_id'),
                        'question': question,
                        'template': template_bucket(question),
                        'ground_truth': sample.get('answer'),
                        'ground_truth_yn': gt,
                        'variants': {
                            name: {
                                **probs,
                                'prediction': pred_from_probs(probs),
                                'is_correct': judge_correct(pred_from_probs(probs), gt, 'closed') if gt else None,
                            }
                            for name, probs in variants.items()
                        },
                    }
                except Exception as exc:
                    row = {
                        'version': 'shortcut_binary_probs_v1',
                        'status': 'error',
                        'model': model_name,
                        'dataset': dataset,
                        'qtype': 'closed',
                        'dataset_index': idx,
                        'sample_id': sample_id_for(idx, sample),
                        'question_id': sample.get('question_id'),
                        'question': sample.get('question'),
                        'error': repr(exc),
                    }
                append_jsonl(out_path, row)
                print(f'[{model_name}/{dataset}/closed/binary-shortcut] {idx + 1}/{len(ds)} {row["status"]}', flush=True)
            summarize_binary_file(out_path, out_dir / 'summary' / f'{model_name}__{dataset}__closed.json')
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def summarize_binary_file(path: Path, out_json: Path) -> dict[str, Any]:
    rows = [r for r in load_jsonl(path) if r.get('status') == 'ok']
    variants = ('real', 'textonly', 'black', 'random_image')
    summary: dict[str, Any] = {'file': str(path), 'n': len(rows), 'variants': {}, 'agreement': {}, 'templates': {}}
    for name in variants:
        vals = [r['variants'][name] for r in rows if name in (r.get('variants') or {})]
        correct = [v.get('is_correct') for v in vals if isinstance(v.get('is_correct'), bool)]
        pmax = [max(float(v.get('p_yes', 0.0)), float(v.get('p_no', 0.0))) for v in vals]
        yes_rate = [1.0 if v.get('prediction') == 'yes' else 0.0 for v in vals]
        summary['variants'][name] = {
            'n': len(vals),
            'accuracy': float(np.mean(correct)) if correct else None,
            'mean_binary_confidence': float(np.mean(pmax)) if pmax else None,
            'yes_rate': float(np.mean(yes_rate)) if yes_rate else None,
        }
    real_preds = [r['variants']['real']['prediction'] for r in rows if 'real' in r.get('variants', {})]
    for name in ('textonly', 'black', 'random_image'):
        pairs = [(r['variants']['real']['prediction'], r['variants'][name]['prediction']) for r in rows if 'real' in r.get('variants', {}) and name in r.get('variants', {})]
        summary['agreement'][f'real_vs_{name}'] = float(np.mean([a == b for a, b in pairs])) if pairs else None
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[row.get('template', 'unknown')].append(row)
    for bucket, bucket_rows in buckets.items():
        summary['templates'][bucket] = {
            'n': len(bucket_rows),
            'gt_yes_rate': float(np.mean([1.0 if r.get('ground_truth_yn') == 'yes' else 0.0 for r in bucket_rows])),
            'real_yes_rate': float(np.mean([1.0 if r['variants']['real']['prediction'] == 'yes' else 0.0 for r in bucket_rows])),
            'textonly_yes_rate': float(np.mean([1.0 if r['variants']['textonly']['prediction'] == 'yes' else 0.0 for r in bucket_rows])),
            'real_textonly_agreement': float(np.mean([r['variants']['real']['prediction'] == r['variants']['textonly']['prediction'] for r in bucket_rows])),
        }
    write_json(out_json, summary)
    write_binary_markdown(out_json.with_suffix('.md'), summary)
    return summary


def write_binary_markdown(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        '# Binary Shortcut Probe',
        '',
        f"- source: `{summary.get('file')}`",
        f"- n: {summary.get('n')}",
        '',
        '| variant | accuracy | mean binary confidence | yes rate |',
        '|---|---:|---:|---:|',
    ]
    for name, row in (summary.get('variants') or {}).items():
        lines.append(f"| {name} | {fmt(row.get('accuracy'))} | {fmt(row.get('mean_binary_confidence'))} | {fmt(row.get('yes_rate'))} |")
    lines.extend(['', '| agreement | value |', '|---|---:|'])
    for name, value in (summary.get('agreement') or {}).items():
        lines.append(f'| {name} | {fmt(value)} |')
    lines.extend(['', '| template | n | GT yes rate | real yes rate | text-only yes rate | real/text-only agreement |', '|---|---:|---:|---:|---:|---:|'])
    for bucket, row in sorted((summary.get('templates') or {}).items(), key=lambda kv: -int(kv[1].get('n', 0))):
        lines.append(f"| {bucket} | {row.get('n')} | {fmt(row.get('gt_yes_rate'))} | {fmt(row.get('real_yes_rate'))} | {fmt(row.get('textonly_yes_rate'))} | {fmt(row.get('real_textonly_agreement'))} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('template-prior')
    p.add_argument('--datasets', nargs='+', default=list(DATASETS))
    p.add_argument('--output-dir', default=str(DEFAULT_OUT))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-samples', type=int)

    p = sub.add_parser('high-conf-wrong')
    p.add_argument('--models', nargs='+', default=['qwen3vl_8b', 'internvl3_8b_local', 'lingshu_7b', 'medgemma_4b', 'medvlm_r1', 'qwen3vl_2b'])
    p.add_argument('--datasets', nargs='+', default=list(DATASETS))
    p.add_argument('--greedy-dir', default=str(BENCH_ROOT / '02_inputs/greedy_answers'))
    p.add_argument('--label-dir', default=str(BENCH_ROOT / '02_inputs/labels'))
    p.add_argument('--hac-dir', default=str(BENCH_ROOT / '03_runs/hac/raw'))
    p.add_argument('--cas-closed-dir', default=str(BENCH_ROOT / '03_runs/cas_closed/features'))
    p.add_argument('--output-dir', default=str(DEFAULT_OUT))
    p.add_argument('--hac-conf-threshold', type=float, default=0.9)
    p.add_argument('--real-conf-threshold', type=float, default=0.9)

    p = sub.add_parser('binary-shortcut')
    p.add_argument('--models', nargs='+', required=True)
    p.add_argument('--datasets', nargs='+', required=True)
    p.add_argument('--output-dir', default=str(DEFAULT_OUT))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-samples', type=int)
    p.add_argument('--quantization')
    p.add_argument('--device-map', default='auto')
    p.add_argument('--no-flash-attention', action='store_true')
    p.add_argument('--resume', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == 'template-prior':
        template_prior(args)
    elif args.cmd == 'high-conf-wrong':
        high_conf_wrong(args)
    elif args.cmd == 'binary-shortcut':
        binary_shortcut(args)
    else:
        raise ValueError(args.cmd)


if __name__ == '__main__':
    main()
