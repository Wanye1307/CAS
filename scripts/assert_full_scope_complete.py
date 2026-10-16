#!/usr/bin/env python3
"""Fail unless all expected full-scope benchmark row files are complete."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('qwen3vl_8b', 'internvl3_8b_local', 'lingshu_7b', 'medgemma_4b', 'medvlm_r1')
DATASETS = ('vqa_rad', 'slake', 'vqa_med_2019')
QTYPES = ('closed', 'open')
EXPECTED = {
    ('vqa_rad', 'closed'): 251,
    ('vqa_rad', 'open'): 200,
    ('slake', 'closed'): 355,
    ('slake', 'open'): 706,
    ('vqa_med_2019', 'closed'): 64,
    ('vqa_med_2019', 'open'): 436,
}
METHOD_DIRS = {
    'greedy': ROOT / '02_inputs/greedy_answers',
    'labels': ROOT / '02_inputs/labels',
    'hac': ROOT / '03_runs/hac/raw',
    'cas_closed': ROOT / '03_runs/cas_closed/features',
    'cas_open': ROOT / '03_runs/cas_open/features',
}

def counts(path: Path) -> tuple[int, int, int]:
    total = ok = err = 0
    if not path.exists():
        return total, ok, err
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                row: dict[str, Any] = json.loads(line)
            except Exception:
                err += 1
                continue
            status = row.get('status')
            if status == 'ok' or ('is_correct' in row and status is None):
                ok += 1
            elif status == 'error':
                err += 1
    return total, ok, err

failures = []
for model in MODELS:
    for dataset in DATASETS:
        for qtype in QTYPES:
            expected = EXPECTED[(dataset, qtype)]
            methods = ['greedy', 'labels', 'hac', 'cas_open' if qtype == 'open' else 'cas_closed']
            for method in methods:
                path = METHOD_DIRS[method] / f'{model}__{dataset}__{qtype}.jsonl'
                total, ok, err = counts(path)
                if ok != expected or err != 0:
                    failures.append(f'{model}/{dataset}/{qtype}/{method}: expected_ok={expected} rows={total} ok={ok} err={err}')
if failures:
    print('INCOMPLETE')
    for item in failures[:80]:
        print(item)
    raise SystemExit(1)
print('complete: all full-scope files have expected ok row counts and zero error rows')
