#!/usr/bin/env python3
"""Shared helpers for the greedy HAC/CAS benchmark.

These helpers intentionally live inside the benchmark directory so the new
protocol can evolve without changing legacy repro scripts.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

BENCH_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BENCH_ROOT
DATA_ROOT = Path(os.environ.get('CAS_DATA_ROOT', str(BENCH_ROOT / 'data'))).expanduser().resolve()

for path in (REPO_ROOT, REPO_ROOT / 'src', REPO_ROOT / 'scripts'):
    path_s = str(path)
    if path_s not in sys.path:
        sys.path.insert(0, path_s)

DATA_PATHS = {
    'slake': DATA_ROOT / 'Slake1.0',
    'vqa_med_2019': DATA_ROOT / 'vqa_med_2019/VQAMed2019Test',
    'vqa_med_2020': DATA_ROOT / 'vqa_med_2020/VQA-TestSet-ReferenceAnswers-VQAMed2020-Task1',
    'vqa_med_2021': DATA_ROOT / 'vqa_med_2021/Task1-VQA-2021-TestSet-w-GroundTruth',
}

EPS = 1e-12


def sanitize(value: Any) -> Any:
    try:
        import torch
        if torch.is_tensor(value):
            if value.numel() == 1:
                return sanitize(value.detach().cpu().item())
            return sanitize(value.detach().cpu().tolist())
    except Exception:
        pass
    if isinstance(value, Mapping):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    if isinstance(value, np.ndarray):
        return sanitize(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(sanitize(row), ensure_ascii=False) + '\n')


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8', buffering=1) as f:
        f.write(json.dumps(sanitize(row), ensure_ascii=False) + '\n')


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize(payload), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def output_path(base: Path, model: str, dataset: str, qtype: str, suffix: str = '.jsonl') -> Path:
    return base / f'{model}__{dataset}__{qtype}{suffix}'


def normalize_yes_no(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower().rstrip('.')
    text = re.sub(r'^answer\s*:\s*', '', text).strip().rstrip('.')
    if text in {'yes', 'y'} or text.startswith('yes'):
        return 'yes'
    if text in {'no', 'n'} or text.startswith('no'):
        return 'no'
    if 'yes' in text and 'no' not in text:
        return 'yes'
    if 'no' in text and 'yes' not in text:
        return 'no'
    return None


def strip_answer_prefix(text: Any) -> str:
    s = '' if text is None else str(text)
    s = s.strip()
    match = re.search(r'answer\s*:\s*(.+)$', s, flags=re.IGNORECASE | re.DOTALL)
    if match:
        s = match.group(1).strip()
    s = re.sub(r'^(assistant|model)\s*:\s*', '', s, flags=re.IGNORECASE).strip()
    return s.strip().strip('`').strip().rstrip('.')


def normalize_open_answer(value: Any) -> str:
    return strip_answer_prefix(value).lower().strip()


def parse_generation(raw_generation: Any, qtype: str) -> tuple[str, Optional[str]]:
    raw = '' if raw_generation is None else str(raw_generation)
    try:
        from medvlm.confidence.sampling import parse_answer_text
        parsed = parse_answer_text(raw)
    except Exception:
        parsed = None
    answer_text = strip_answer_prefix(parsed if parsed else raw)
    if qtype == 'closed':
        return answer_text, normalize_yes_no(answer_text)
    return answer_text, normalize_open_answer(answer_text)


def judge_correct(prediction: Any, ground_truth: Any, qtype: str) -> Optional[bool]:
    if ground_truth is None:
        return None
    if qtype == 'closed':
        pred = normalize_yes_no(prediction)
        gt = normalize_yes_no(ground_truth)
        if pred is None or gt is None:
            return False
        return pred == gt
    pred = normalize_open_answer(prediction)
    gt = normalize_open_answer(ground_truth)
    if not pred or not gt:
        return False
    return pred == gt


def sample_id_for(index: int, sample: Mapping[str, Any]) -> str:
    qid = sample.get('question_id')
    if qid is not None and str(qid) != '':
        return f'qid:{qid}'
    image_id = sample.get('image_id')
    if image_id is not None and str(image_id) != '':
        return f'image:{image_id}:idx:{index}'
    return f'idx:{index}'


def row_sample_id(row: Mapping[str, Any]) -> str:
    sid = row.get('sample_id')
    if sid is not None and str(sid) != '':
        return str(sid)
    if row.get('question_id') is not None:
        return f'qid:{row.get("question_id")}'
    return f'idx:{row.get("dataset_index", row.get("source_index", "NA"))}'


def load_labels(path: Path) -> dict[str, dict[str, Any]]:
    return {row_sample_id(row): row for row in load_jsonl(path)}


def apply_label(row: dict[str, Any], labels: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    label = labels.get(row_sample_id(row))
    out = dict(row)
    if label is not None:
        out['is_correct'] = label.get('is_correct')
        out['label_source_path'] = label.get('source_path') or str(label.get('label_source_path') or '')
        out['judge_method'] = label.get('judge_method')
    return out


def load_dataset(dataset: str, qtype: str, seed: int, max_samples: Optional[int] = None):
    import medvlm
    kwargs: dict[str, Any] = {'split': 'test', 'question_type': qtype, 'seed': seed}
    if dataset in DATA_PATHS:
        kwargs['data_path'] = str(DATA_PATHS[dataset])
    if max_samples is not None:
        kwargs['subsample_size'] = int(max_samples)
    return medvlm.load_dataset(dataset, **kwargs)


def set_seed(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def done_indices(path: Path) -> set[int]:
    out: set[int] = set()
    for row in load_jsonl(path):
        if row.get('status') == 'ok' and row.get('dataset_index') is not None:
            out.add(int(row['dataset_index']))
    return out


def target_for_row(row: Mapping[str, Any], qtype: str) -> tuple[str, Optional[str]]:
    if qtype == 'closed':
        pred = normalize_yes_no(row.get('greedy_prediction') or row.get('target_prediction') or row.get('target_answer'))
        return pred or '', pred
    target = row.get('target_answer') or row.get('greedy_answer') or row.get('greedy_text') or row.get('raw_generation')
    return normalize_open_answer(target), None


def token_ids_for_answer(processor: Any, answer: str) -> list[int]:
    tokenizer = getattr(processor, 'tokenizer', processor)
    ids: list[int] = []
    if hasattr(tokenizer, 'encode'):
        try:
            ids = list(tokenizer.encode(answer, add_special_tokens=False))
        except TypeError:
            ids = list(tokenizer.encode(answer))
    if not ids and callable(tokenizer):
        encoded = tokenizer(answer, add_special_tokens=False)
        ids = list(encoded.get('input_ids', []))
    stop_ids = set()
    for attr in ('eos_token_id', 'pad_token_id', 'bos_token_id'):
        token_id = getattr(tokenizer, attr, None)
        if token_id is None:
            continue
        if isinstance(token_id, Sequence) and not isinstance(token_id, (str, bytes)):
            stop_ids.update(int(x) for x in token_id if x is not None)
        else:
            stop_ids.add(int(token_id))
    return [int(x) for x in ids if int(x) not in stop_ids]
