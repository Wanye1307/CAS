#!/usr/bin/env python3
"""Compute HAC/VASE rows anchored to shared greedy answers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_common import (  # noqa: E402
    BENCH_ROOT,
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

from medvlm.confidence import hedge as hedge_mod  # noqa: E402
from medvlm.confidence.sampling import BASE_SAMPLING_PROMPT  # noqa: E402
import medvlm  # noqa: E402


def cluster_fn_for(qtype: str, threshold: float) -> Callable[[list[str]], list[int]]:
    if qtype == 'closed':
        return hedge_mod.cluster_by_yesno
    return lambda answers: hedge_mod.cluster_by_embedding(answers, threshold=threshold)


def confidence_from_clusters(cluster_ids: list[int], n_samples: int) -> float:
    if not cluster_ids or n_samples <= 0:
        return 0.0
    target_cluster = int(cluster_ids[0])
    sample_clusters = [int(x) for x in cluster_ids[1:1 + n_samples]]
    return float(sum(1 for x in sample_clusters if x == target_cluster) / n_samples)


def run(args: argparse.Namespace) -> None:
    greedy_dir = Path(args.greedy_dir)
    label_dir = Path(args.label_dir)
    out_dir = Path(args.output_dir)
    hb = hedge_mod._load_hedge_bench()
    for model_name in args.models:
        print(f'Loading model {model_name}', flush=True)
        model, processor, model_config = medvlm.load_model(
            model_name,
            quantization=args.quantization,
            use_flash_attention=not args.no_flash_attention,
            device_map=args.device_map,
        )
        for dataset in args.datasets:
            for qtype in args.qtypes:
                greedy_path = output_path(greedy_dir, model_name, dataset, qtype)
                label_path = output_path(label_dir, model_name, dataset, qtype)
                labels = load_labels(label_path)
                greedy_rows = {int(r['dataset_index']): apply_label(r, labels) for r in load_jsonl(greedy_path) if r.get('status') == 'ok'}
                ds = load_dataset(dataset, qtype, args.seed, args.max_samples)
                out_path = output_path(out_dir, model_name, dataset, qtype)
                done = done_indices(out_path) if args.resume else set()
                max_new_tokens = args.max_new_tokens_closed if qtype == 'closed' else args.max_new_tokens_open
                for idx, sample in enumerate(ds):
                    if idx in done:
                        continue
                    g = greedy_rows.get(idx)
                    if not g:
                        continue
                    sample_seed = args.seed * 1_000_003 + idx
                    set_seed(sample_seed)
                    try:
                        question = str(sample['question'])
                        target_answer = str(g.get('target_prediction') if qtype == 'closed' else g.get('target_answer') or '')
                        if qtype == 'closed':
                            target_answer = normalize_yes_no(target_answer) or ''
                        inputs = hedge_mod._format_single(
                            processor,
                            model_config.model_family,
                            sample['image'],
                            question,
                            BASE_SAMPLING_PROMPT,
                        )
                        orig_ans, orig_lps = hedge_mod._generate(
                            model,
                            processor,
                            inputs,
                            num_return_sequences=args.n_samples,
                            temperature=args.temperature,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                        )
                        distorted = hedge_mod._distort(sample['image'], args.n_samples, hb['distort_image'])
                        dist_ans, dist_lps = hedge_mod._generate_distorted_batch(
                            model,
                            processor,
                            distorted,
                            question,
                            BASE_SAMPLING_PROMPT,
                            model_config.model_family,
                            args.temperature,
                            max_new_tokens,
                            args.batch_size,
                        )
                        cfn = cluster_fn_for(qtype, args.cluster_threshold)
                        all_answers = [target_answer] + list(orig_ans) + list(dist_ans)
                        cluster_ids = [int(x) for x in cfn(all_answers)]
                        ent_clean, dist_clean = hb['sentence_semantic_entropy'](orig_lps, cluster_ids[1:1 + args.n_samples])
                        _ent_noisy, dist_noisy = hb['sentence_semantic_entropy'](dist_lps, cluster_ids[1 + args.n_samples:])
                        vase = hb['vase'](args.n_samples, cluster_ids, dist_clean, dist_noisy, args.alpha)
                        radflag = hb['radflag'](cluster_ids, args.n_samples)
                        row = {
                            'version': 'hac_vase_from_shared_greedy_v1',
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
                            'target_answer': target_answer,
                            'target_prediction': g.get('target_prediction'),
                            'is_correct': g.get('is_correct'),
                            'n_samples': args.n_samples,
                            'temperature': args.temperature,
                            'alpha': args.alpha,
                            'cluster_threshold': args.cluster_threshold,
                            'cluster_ids': cluster_ids,
                            'hac_confidence_n10': confidence_from_clusters(cluster_ids, args.n_samples),
                            'VASE': float(vase),
                            'SE': float(ent_clean),
                            'RadFlag': float(radflag),
                            'original_high_temp': list(orig_ans),
                            'original_logprobs': [float(x) for x in orig_lps],
                            'distorted_high_temp': list(dist_ans),
                            'distorted_logprobs': [float(x) for x in dist_lps],
                            'seed': sample_seed,
                        }
                    except Exception as exc:
                        row = {
                            'version': 'hac_vase_from_shared_greedy_v1',
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
                    print(f'[{model_name}/{dataset}/{qtype}] {idx + 1}/{len(ds)} {row["status"]}', flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument('--qtypes', nargs='+', choices=['closed', 'open'], required=True)
    parser.add_argument('--greedy-dir', default=str(BENCH_ROOT / '02_inputs/greedy_answers'))
    parser.add_argument('--label-dir', default=str(BENCH_ROOT / '02_inputs/labels'))
    parser.add_argument('--output-dir', default=str(BENCH_ROOT / '03_runs/hac/raw'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--n-samples', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--cluster-threshold', type=float, default=0.85)
    parser.add_argument('--batch-size', type=int, default=5)
    parser.add_argument('--max-new-tokens-closed', type=int, default=16)
    parser.add_argument('--max-new-tokens-open', type=int, default=64)
    parser.add_argument('--quantization', default=None)
    parser.add_argument('--device-map', default='auto')
    parser.add_argument('--no-flash-attention', action='store_true')
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())
