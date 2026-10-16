#!/usr/bin/env python3
"""Compute closed CAS features anchored to shared VASE/HEDGE greedy answers."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_common import (  # noqa: E402
    BENCH_ROOT,
    EPS,
    append_jsonl,
    apply_label,
    done_indices,
    load_dataset,
    load_jsonl,
    load_labels,
    normalize_yes_no,
    output_path,
    row_sample_id,
    set_seed,
)

import medvlm  # noqa: E402
from cas_runtime import make_baseline, run_binary  # noqa: E402

BASELINES = ('textonly', 'black', 'noise', 'blur')


def safe_log(value: float) -> float:
    return math.log(max(float(value), EPS))


def prob_for(probs: Mapping[str, Any], answer: str) -> float:
    return float(probs['p_yes'] if answer == 'yes' else probs['p_no'])


def logp_for(probs: Mapping[str, Any], answer: str) -> float:
    key = 'log_p_yes' if answer == 'yes' else 'log_p_no'
    if key in probs:
        return float(probs[key])
    return safe_log(prob_for(probs, answer))


def compute_features(original: Mapping[str, Any], baselines: Mapping[str, Mapping[str, Any]], pred: str) -> dict[str, float]:
    opp = 'no' if pred == 'yes' else 'yes'
    real_pred = logp_for(original, pred)
    real_opp = logp_for(original, opp)
    pred_risks: dict[str, float] = {}
    opp_risks: dict[str, float] = {}
    cf_supports: list[float] = []
    same_pred: list[float] = []
    for name in BASELINES:
        cf = baselines[name]
        short = 'text' if name == 'textonly' else name
        pred_cf = logp_for(cf, pred)
        opp_cf = logp_for(cf, opp)
        pred_risks[short] = pred_cf - real_pred
        opp_risks[short] = (real_opp - opp_cf) - (real_pred - pred_cf)
        cf_supports.append(prob_for(cf, pred))
        cf_pred = 'yes' if float(cf.get('p_yes', 0.0)) >= float(cf.get('p_no', 0.0)) else 'no'
        same_pred.append(1.0 if cf_pred == pred else 0.0)
    out: dict[str, float] = {}
    for short in ('text', 'black', 'noise', 'blur'):
        out[f'cas_pred_{short}'] = float(pred_risks[short])
        out[f'cas_opp_{short}'] = float(opp_risks[short])
    pred_values = list(pred_risks.values())
    opp_values = list(opp_risks.values())
    out.update({
        'cas_pred_mean': float(np.mean(pred_values)),
        'cas_pred_max': float(np.max(pred_values)),
        'cas_pred_min': float(np.min(pred_values)),
        'cas_opp_mean': float(np.mean(opp_values)),
        'cas_opp_max': float(np.max(opp_values)),
        'cas_opp_min': float(np.min(opp_values)),
        'cas_cf_support_mean': float(np.mean(cf_supports)),
        'cas_cf_support_max': float(np.max(cf_supports)),
        'cas_same_pred_frac': float(np.mean(same_pred)),
    })
    return out


def run(args: argparse.Namespace) -> None:
    greedy_dir = Path(args.greedy_dir)
    label_dir = Path(args.label_dir)
    out_dir = Path(args.output_dir)
    for model_name in args.models:
        print(f'Loading model {model_name}', flush=True)
        model, processor, model_config = medvlm.load_model(
            model_name,
            quantization=args.quantization,
            use_flash_attention=not args.no_flash_attention,
            device_map=args.device_map,
        )
        for dataset in args.datasets:
            qtype = 'closed'
            greedy_path = output_path(greedy_dir, model_name, dataset, qtype)
            label_path = output_path(label_dir, model_name, dataset, qtype)
            labels = load_labels(label_path)
            greedy_rows = {int(r['dataset_index']): apply_label(r, labels) for r in load_jsonl(greedy_path) if r.get('status') == 'ok'}
            ds = load_dataset(dataset, qtype, args.seed, args.max_samples)
            out_path = output_path(out_dir, model_name, dataset, qtype)
            done = done_indices(out_path) if args.resume else set()
            for idx, sample in enumerate(ds):
                if idx in done:
                    continue
                g = greedy_rows.get(idx)
                if not g:
                    continue
                sample_seed = args.seed * 1_000_003 + idx
                set_seed(sample_seed)
                try:
                    target = normalize_yes_no(g.get('target_prediction') or g.get('target_answer'))
                    if target not in {'yes', 'no'}:
                        raise ValueError(f'closed target is not yes/no: {g.get("target_answer")!r}')
                    question = str(sample['question'])
                    original = run_binary(model, processor, model_config, sample['image'], question)
                    baselines: dict[str, Mapping[str, Any]] = {}
                    for offset, name in enumerate(BASELINES):
                        image = None if name == 'textonly' else make_baseline(name, sample['image'], seed=sample_seed + offset)
                        baselines[name] = run_binary(model, processor, model_config, image, question)
                    row = {
                        'version': 'cas_closed_from_shared_greedy_v1',
                        'status': 'ok',
                        'model': model_name,
                        'dataset': dataset,
                        'qtype': qtype,
                        'dataset_index': idx,
                        'sample_id': row_sample_id(g),
                        'question_id': g.get('question_id'),
                        'image_id': g.get('image_id'),
                        'question': question,
                        'ground_truth': g.get('ground_truth'),
                        'target_answer': target,
                        'target_prediction': target,
                        'is_correct': g.get('is_correct'),
                        'original': original,
                        'baselines': baselines,
                        'features': compute_features(original, baselines, target),
                        'seed': sample_seed,
                    }
                except Exception as exc:
                    row = {
                        'version': 'cas_closed_from_shared_greedy_v1',
                        'status': 'error',
                        'model': model_name,
                        'dataset': dataset,
                        'qtype': qtype,
                        'dataset_index': idx,
                        'sample_id': row_sample_id(g),
                        'question_id': g.get('question_id'),
                        'error': repr(exc),
                    }
                append_jsonl(out_path, row)
                print(f'[{model_name}/{dataset}/closed] {idx + 1}/{len(ds)} {row["status"]}', flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument('--greedy-dir', default=str(BENCH_ROOT / '02_inputs/greedy_answers'))
    parser.add_argument('--label-dir', default=str(BENCH_ROOT / '02_inputs/labels'))
    parser.add_argument('--output-dir', default=str(BENCH_ROOT / '03_runs/cas_closed/features'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--quantization', default=None)
    parser.add_argument('--device-map', default='auto')
    parser.add_argument('--no-flash-attention', action='store_true')
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())
