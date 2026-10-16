#!/usr/bin/env python3
"""Create normalized correctness labels for shared greedy answers.

Closed labels are exact yes/no. Open labels use normalized exact matching;
substring containment is intentionally not used. If you later use an LLM judge,
provide a JSONL file with matching sample_id and is_correct via --external-labels.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_common import (  # noqa: E402
    BENCH_ROOT,
    judge_correct,
    load_jsonl,
    load_labels,
    output_path,
    row_sample_id,
    write_jsonl,
)


def label_rows(greedy_rows: list[dict[str, Any]], external: dict[str, dict[str, Any]], judge_method: str) -> list[dict[str, Any]]:
    out = []
    for row in greedy_rows:
        if row.get('status') != 'ok':
            continue
        sid = row_sample_id(row)
        ext = external.get(sid)
        if ext is not None and isinstance(ext.get('is_correct'), bool):
            is_correct = bool(ext['is_correct'])
            method = str(ext.get('judge_method') or 'external')
            source_path = str(ext.get('source_path') or ext.get('label_source_path') or '')
        else:
            is_correct = judge_correct(row.get('target_prediction') or row.get('target_answer'), row.get('ground_truth'), str(row['qtype']))
            method = judge_method
            source_path = str(row.get('source_path') or '')
        out.append({
            'version': 'shared_greedy_label_v1',
            'model': row.get('model'),
            'dataset': row.get('dataset'),
            'qtype': row.get('qtype'),
            'dataset_index': row.get('dataset_index'),
            'sample_id': sid,
            'question_id': row.get('question_id'),
            'image_id': row.get('image_id'),
            'question': row.get('question'),
            'ground_truth': row.get('ground_truth'),
            'prediction': row.get('target_prediction') or row.get('target_answer'),
            'target_answer': row.get('target_answer'),
            'target_prediction': row.get('target_prediction'),
            'is_correct': is_correct,
            'judge_model': 'deterministic' if ext is None else ext.get('judge_model', 'external'),
            'judge_method': method,
            'source_path': source_path,
            'source_index': row.get('dataset_index'),
        })
    return out


def run(args: argparse.Namespace) -> None:
    greedy_dir = Path(args.greedy_dir)
    out_dir = Path(args.output_dir)
    external = load_labels(Path(args.external_labels)) if args.external_labels else {}
    for path in sorted(greedy_dir.glob('*.jsonl')):
        parts = path.stem.split('__')
        if len(parts) != 3:
            continue
        model, dataset, qtype = parts
        if args.models and model not in set(args.models):
            continue
        if args.datasets and dataset not in set(args.datasets):
            continue
        if args.qtypes and qtype not in set(args.qtypes):
            continue
        rows = label_rows(load_jsonl(path), external, args.judge_method)
        write_jsonl(output_path(out_dir, model, dataset, qtype), rows)
        n = len(rows)
        correct = sum(1 for r in rows if r.get('is_correct') is True)
        print(f'{path.name}: wrote {n} labels, correct={correct}, accuracy={(correct / n if n else 0):.4f}', flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--greedy-dir', default=str(BENCH_ROOT / '02_inputs/greedy_answers'))
    parser.add_argument('--output-dir', default=str(BENCH_ROOT / '02_inputs/labels'))
    parser.add_argument('--external-labels', help='Optional JSONL with sample_id and is_correct')
    parser.add_argument('--models', nargs='+', help='Optional model keys to label')
    parser.add_argument('--datasets', nargs='+', help='Optional dataset keys to label')
    parser.add_argument('--qtypes', nargs='+', choices=['closed', 'open'], help='Optional question types to label')
    parser.add_argument('--judge-method', default='auto_closed_exact_open_normalized_exact')
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())
