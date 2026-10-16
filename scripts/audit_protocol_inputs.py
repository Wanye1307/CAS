#!/usr/bin/env python3
"""Audit that HAC and CAS rows follow the shared-greedy protocol."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_common import BENCH_ROOT, load_jsonl, output_path, row_sample_id, write_json  # noqa: E402


def audit(greedy_file: Path, label_file: Path, hac_file: Path | None, cas_file: Path | None) -> dict[str, Any]:
    greedy = {row_sample_id(r): r for r in load_jsonl(greedy_file) if r.get('status') == 'ok'}
    labels = {row_sample_id(r): r for r in load_jsonl(label_file)}
    out: dict[str, Any] = {
        'greedy_file': str(greedy_file),
        'label_file': str(label_file),
        'n_greedy': len(greedy),
        'n_labels': len(labels),
        'missing_labels': sorted(set(greedy) - set(labels))[:20],
        'target_mismatches': [],
        'label_mismatches': [],
    }
    for name, path in [('hac', hac_file), ('cas', cas_file)]:
        if path is None or not path.exists():
            out[f'{name}_available'] = False
            continue
        rows = {row_sample_id(r): r for r in load_jsonl(path) if r.get('status') == 'ok'}
        out[f'n_{name}'] = len(rows)
        out[f'{name}_available'] = True
        out[f'{name}_missing_greedy'] = sorted(set(rows) - set(greedy))[:20]
        for sid in sorted(set(rows) & set(greedy)):
            r = rows[sid]
            g = greedy[sid]
            if str(r.get('target_answer') or '').strip().lower() != str(g.get('target_prediction') or g.get('target_answer') or '').strip().lower():
                out['target_mismatches'].append({'source': name, 'sample_id': sid, 'greedy': g.get('target_answer'), 'row': r.get('target_answer')})
                if len(out['target_mismatches']) >= 20:
                    break
            lab = labels.get(sid)
            if lab is not None and isinstance(r.get('is_correct'), bool) and bool(r['is_correct']) != bool(lab.get('is_correct')):
                out['label_mismatches'].append({'source': name, 'sample_id': sid})
                if len(out['label_mismatches']) >= 20:
                    break
    out['passed'] = not out['missing_labels'] and not out['target_mismatches'] and not out['label_mismatches']
    return out


def run(args: argparse.Namespace) -> None:
    greedy_file = Path(args.greedy_file) if args.greedy_file else output_path(Path(args.greedy_dir), args.model, args.dataset, args.qtype)
    label_file = Path(args.label_file) if args.label_file else output_path(Path(args.label_dir), args.model, args.dataset, args.qtype)
    hac_file = Path(args.hac_file) if args.hac_file else output_path(Path(args.hac_dir), args.model, args.dataset, args.qtype)
    cas_base = Path(args.cas_open_dir if args.qtype == 'open' else args.cas_closed_dir)
    cas_file = Path(args.cas_file) if args.cas_file else output_path(cas_base, args.model, args.dataset, args.qtype)
    result = audit(greedy_file, label_file, hac_file, cas_file)
    write_json(Path(args.output), result)
    print(f'passed={result["passed"]}; wrote {args.output}', flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model')
    parser.add_argument('--dataset')
    parser.add_argument('--qtype', choices=['closed', 'open'], default='closed')
    parser.add_argument('--greedy-file')
    parser.add_argument('--label-file')
    parser.add_argument('--hac-file')
    parser.add_argument('--cas-file')
    parser.add_argument('--greedy-dir', default=str(BENCH_ROOT / '02_inputs/greedy_answers'))
    parser.add_argument('--label-dir', default=str(BENCH_ROOT / '02_inputs/labels'))
    parser.add_argument('--hac-dir', default=str(BENCH_ROOT / '03_runs/hac/raw'))
    parser.add_argument('--cas-closed-dir', default=str(BENCH_ROOT / '03_runs/cas_closed/features'))
    parser.add_argument('--cas-open-dir', default=str(BENCH_ROOT / '03_runs/cas_open/features'))
    parser.add_argument('--output', default=str(BENCH_ROOT / '05_audits/protocol_audit.json'))
    args = parser.parse_args()
    if not args.greedy_file and (not args.model or not args.dataset):
        parser.error('--model and --dataset are required unless explicit files are provided')
    return args


if __name__ == '__main__':
    run(parse_args())
