#!/usr/bin/env python3
"""Compute open CAS features by teacher-forcing shared greedy answers."""

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
    append_jsonl,
    apply_label,
    done_indices,
    finite,
    load_dataset,
    load_jsonl,
    load_labels,
    output_path,
    row_sample_id,
    set_seed,
    token_ids_for_answer,
)

import medvlm  # noqa: E402
from confidence.grounding import OPEN_GROUNDING_PROMPT  # noqa: E402
from cas_open_likelihood import answer_token_likelihood, baseline_image  # noqa: E402

BASELINES = ('text', 'black', 'noise', 'blur')


def likelihood_valid(stats: Mapping[str, Any]) -> bool:
    return int(stats.get('token_count') or 0) > 0 and finite(stats.get('mean_logprob'))


def compute_features(real: Mapping[str, Any], cf: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    token_count = float(real.get('token_count') or 0.0)
    base = {'token_count': token_count, 'token_count_log': math.log1p(max(token_count, 0.0)), 'likelihood_valid': 0.0}
    if not likelihood_valid(real) or any(name not in cf or not likelihood_valid(cf[name]) for name in BASELINES):
        return base
    real_logprob = float(real['mean_logprob'])
    cas_values = {name: float(cf[name]['mean_logprob']) - real_logprob for name in BASELINES}
    positive = [max(0.0, value) for value in cas_values.values()]
    out = {
        **base,
        'likelihood_valid': 1.0,
        'real_confidence': float(real.get('confidence', 0.0)),
        'real_nll': float(real.get('nll', -real_logprob)),
        'real_mean_logprob': real_logprob,
        'real_sum_logprob': float(real.get('sum_logprob', real_logprob * token_count)),
        'cf_logprob_max': float(np.max([float(cf[name]['mean_logprob']) for name in BASELINES])),
        'cf_logprob_mean': float(np.mean([float(cf[name]['mean_logprob']) for name in BASELINES])),
        'cf_confidence_max': float(np.max([float(cf[name].get('confidence', 0.0)) for name in BASELINES])),
        'cf_confidence_mean': float(np.mean([float(cf[name].get('confidence', 0.0)) for name in BASELINES])),
        'cas_text': cas_values['text'],
        'cas_black': cas_values['black'],
        'cas_noise': cas_values['noise'],
        'cas_blur': cas_values['blur'],
        'cas_mean': float(np.mean(list(cas_values.values()))),
        'cas_max': float(np.max(list(cas_values.values()))),
        'cas_positive_mean': float(np.mean(positive)),
        'cas_positive_max': float(np.max(positive)),
    }
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
            qtype = 'open'
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
                    target_answer = str(g.get('target_answer') or '').strip()
                    target_ids = token_ids_for_answer(processor, target_answer)
                    if not target_ids:
                        raise ValueError(f'empty target token ids for answer {target_answer!r}')
                    prompt_text = OPEN_GROUNDING_PROMPT.format(question=sample['question'])
                    real = answer_token_likelihood(
                        model,
                        processor,
                        model_config,
                        sample['image'],
                        prompt_text,
                        target_ids,
                        debug_token_alignment=args.debug_token_alignment,
                        debug_label=f'{model_name}/{dataset}/{idx}',
                        strict_alignment=args.strict_token_alignment,
                    )
                    cf: dict[str, Mapping[str, Any]] = {}
                    for offset, name in enumerate(BASELINES):
                        image = baseline_image(name, sample['image'], seed=sample_seed + offset)
                        cf[name] = answer_token_likelihood(
                            model,
                            processor,
                            model_config,
                            image,
                            prompt_text,
                            target_ids,
                            strict_alignment=args.strict_token_alignment,
                        )
                    row = {
                        'version': 'cas_open_from_shared_greedy_v1',
                        'status': 'ok',
                        'model': model_name,
                        'dataset': dataset,
                        'qtype': qtype,
                        'dataset_index': idx,
                        'sample_id': row_sample_id(g),
                        'question_id': g.get('question_id'),
                        'image_id': g.get('image_id'),
                        'question': sample.get('question'),
                        'ground_truth': g.get('ground_truth'),
                        'target_answer': target_answer,
                        'target_token_ids': target_ids,
                        'is_correct': g.get('is_correct'),
                        'prompt': prompt_text,
                        'likelihood': {'real': real, **cf},
                        'features': compute_features(real, cf),
                        'seed': sample_seed,
                    }
                except Exception as exc:
                    row = {
                        'version': 'cas_open_from_shared_greedy_v1',
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
                print(f'[{model_name}/{dataset}/open] {idx + 1}/{len(ds)} {row["status"]}', flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument('--greedy-dir', default=str(BENCH_ROOT / '02_inputs/greedy_answers'))
    parser.add_argument('--label-dir', default=str(BENCH_ROOT / '02_inputs/labels'))
    parser.add_argument('--output-dir', default=str(BENCH_ROOT / '03_runs/cas_open/features'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--quantization', default=None)
    parser.add_argument('--device-map', default='auto')
    parser.add_argument('--no-flash-attention', action='store_true')
    parser.add_argument('--strict-token-alignment', action='store_true', default=True)
    parser.add_argument('--allow-unsafe-token-alignment', dest='strict_token_alignment', action='store_false')
    parser.add_argument('--debug-token-alignment', action='store_true')
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())
