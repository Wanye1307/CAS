#!/usr/bin/env python3
"""Print row-level progress for the confirmed full HAC/CAS benchmark scope."""
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

def path_for(method: str, model: str, dataset: str, qtype: str) -> Path | None:
    if method == 'cas_closed' and qtype != 'closed':
        return None
    if method == 'cas_open' and qtype != 'open':
        return None
    return METHOD_DIRS[method] / f'{model}__{dataset}__{qtype}.jsonl'

print('| model | dataset | qtype | method | expected | rows | ok | error |')
print('|---|---|---|---|---:|---:|---:|---:|')
for model in MODELS:
    for dataset in DATASETS:
        for qtype in QTYPES:
            expected = EXPECTED[(dataset, qtype)]
            for method in ('greedy', 'labels', 'hac', 'cas_closed', 'cas_open'):
                path = path_for(method, model, dataset, qtype)
                if path is None:
                    continue
                total, ok, err = counts(path)
                print(f'| {model} | {dataset} | {qtype} | {method} | {expected} | {total} | {ok} | {err} |')
