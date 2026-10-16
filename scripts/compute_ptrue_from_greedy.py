#!/usr/bin/env python3
"""Compute protocol-aligned P(True) rows from shared greedy answers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

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
    strip_answer_prefix,
)

from confidence.ptrue import PTRUE_PROMPT, ptrue_from_logits  # noqa: E402
from medvlm.confidence import hedge as hedge_mod  # noqa: E402
import medvlm  # noqa: E402


def model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def move_to_device(inputs: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, 'to') else value for key, value in inputs.items()}


def target_answer_from_greedy(row: Mapping[str, Any], qtype: str) -> str:
    if qtype == 'closed':
        target = normalize_yes_no(row.get('target_prediction') or row.get('target_answer') or row.get('raw_generation'))
        return target or ''
    target = row.get('target_answer') or row.get('greedy_answer') or row.get('raw_generation') or ''
    return strip_answer_prefix(target)


def compute_ptrue(model: Any, processor: Any, model_config: Any, image: Any, question: str, answer: str) -> tuple[float, float, float, float, str]:
    prompt_text = PTRUE_PROMPT.format(question=question, answer=answer)
    inputs = hedge_mod._format_single(
        processor,
        model_config.model_family,
        image,
        prompt_text,
        '{question}',
    )
    inputs = move_to_device(inputs, model_device(model))
    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[:, -1, :]
    p_true, p_false, p_true_raw, p_false_raw = ptrue_from_logits(next_token_logits, processor.tokenizer)
    return p_true, p_false, p_true_raw, p_false_raw, prompt_text


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
            for qtype in args.qtypes:
                greedy_path = output_path(greedy_dir, model_name, dataset, qtype)
                label_path = output_path(label_dir, model_name, dataset, qtype)
                labels = load_labels(label_path)
                greedy_rows = {
                    int(r['dataset_index']): apply_label(r, labels)
                    for r in load_jsonl(greedy_path)
                    if r.get('status') == 'ok'
                }
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
                        question = str(sample['question'])
                        target_answer = target_answer_from_greedy(g, qtype)
                        if not target_answer:
                            raise ValueError('empty shared greedy target answer')
                        p_true, p_false, p_true_raw, p_false_raw, prompt_text = compute_ptrue(
                            model,
                            processor,
                            model_config,
                            sample['image'],
                            question,
                            target_answer,
                        )
                        row = {
                            'version': 'ptrue_from_shared_greedy_v1',
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
                            'confidence': float(p_true),
                            'uncertainty': float(1.0 - p_true),
                            'p_true': float(p_true),
                            'p_false': float(p_false),
                            'p_true_raw': float(p_true_raw),
                            'p_false_raw': float(p_false_raw),
                            'self_eval_prompt': prompt_text,
                            'seed': sample_seed,
                        }
                    except Exception as exc:
                        row = {
                            'version': 'ptrue_from_shared_greedy_v1',
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
                    print(f'[{model_name}/{dataset}/{qtype}/ptrue] {idx + 1}/{len(ds)} {row["status"]}', flush=True)
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
    parser.add_argument('--output-dir', default=str(BENCH_ROOT / '03_runs/ptrue/raw'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--quantization', default=None)
    parser.add_argument('--device-map', default='auto')
    parser.add_argument('--no-flash-attention', action='store_true')
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())
