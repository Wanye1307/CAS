#!/usr/bin/env python3
"""Minimal closed-ended semantic flip and entity-swap probe.

This is a lightweight v3-style probe for the paper's shortcut analysis. It only
tests whether closed yes/no predictions change under:

1. a semantic inversion prompt asking whether the correct answer is "no"; and
2. simple medical entity swaps in the question.

It intentionally does not compute uncertainty metrics, probabilities for tables,
or logistic models. The primary outputs are flip/non-flip rates.
"""

from __future__ import annotations

import argparse
import re
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
    load_dataset,
    load_jsonl,
    normalize_yes_no,
    output_path,
    sample_id_for,
    set_seed,
    write_json,
)

import medvlm  # noqa: E402
from confidence.grounding import CLOSED_GROUNDING_PROMPT, _binary_next_token_probs  # noqa: E402


DATASETS = ("vqa_rad", "slake", "vqa_med_2019")
DEFAULT_OUT = BENCH_ROOT / "06_shortcut_validation/semantic_flip_entity_swap"

MEDICAL_TERMS = [
    "cardiomegaly", "pleural effusion", "pneumothorax", "edema",
    "atelectasis", "consolidation", "infiltrate", "mass", "nodule",
    "fracture", "opacity", "lesion", "tumor", "abnormality",
    "kidneys", "kidney", "lungs", "lung", "heart", "liver", "spleen",
    "brain", "mediastinum", "spine", "ribs", "stomach", "colon",
]

DISTRACTORS = [
    "cardiomegaly", "pleural effusion", "pneumothorax", "edema",
    "atelectasis", "mass", "nodule", "fracture", "kidneys", "lungs",
    "heart", "liver", "spleen", "mediastinum",
]


def fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "NA"


def prompt_for(question: str) -> str:
    return CLOSED_GROUNDING_PROMPT.format(question=question)


def pred_from_probs(probs: Mapping[str, Any]) -> str:
    return "yes" if float(probs.get("p_yes", 0.0)) >= float(probs.get("p_no", 0.0)) else "no"


def score_pred(model: Any, processor: Any, model_config: Any, image: Any, question: str) -> dict[str, Any]:
    probs = _binary_next_token_probs(model, processor, model_config, image, prompt_for(question))
    pred = pred_from_probs(probs)
    return {
        "prediction": pred,
        "p_yes": float(probs.get("p_yes", 0.0)),
        "p_no": float(probs.get("p_no", 0.0)),
    }


def semantic_flip_question(question: str) -> str:
    q = question.strip()
    return f"For the following medical image question, is the correct answer no? Question: {q}"


def find_terms(question: str) -> list[str]:
    q = question.lower()
    found: list[str] = []
    for term in sorted(MEDICAL_TERMS, key=len, reverse=True):
        if re.search(r"\b" + re.escape(term) + r"\b", q):
            found.append(term)
    return found


def entity_swap_questions(question: str, max_swaps: int) -> list[dict[str, Any]]:
    q = question.strip()
    q_lower = q.lower()
    variants: list[dict[str, Any]] = []
    terms = find_terms(q)
    for term in terms[:1]:
        candidates = [d for d in DISTRACTORS if d != term]
        for distractor in candidates[:max_swaps]:
            pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
            swapped = pattern.sub(distractor, q_lower, count=1)
            variants.append({
                "name": f"swap_{term}_to_{distractor}".replace(" ", "_"),
                "question": swapped,
                "term": term,
                "distractor": distractor,
            })
    return variants[:max_swaps]


def score_dataset(args: argparse.Namespace, model_name: str, dataset: str, model: Any, processor: Any, model_config: Any) -> Path:
    ds = load_dataset(dataset, "closed", args.seed, args.max_samples)
    out_path = output_path(Path(args.output_dir) / "raw", model_name, dataset, "closed")
    done = done_indices(out_path) if args.resume else set()
    for idx, sample in enumerate(ds):
        if idx in done:
            continue
        set_seed(args.seed * 1_000_003 + idx)
        try:
            question = str(sample["question"])
            image = sample["image"]
            original = score_pred(model, processor, model_config, image, question)
            original_pred = original["prediction"]
            meta_no_question = semantic_flip_question(question)
            meta_no = score_pred(model, processor, model_config, image, meta_no_question)
            swaps = []
            for variant in entity_swap_questions(question, args.max_swaps):
                scored = score_pred(model, processor, model_config, image, str(variant["question"]))
                swaps.append({
                    **variant,
                    **scored,
                    "flipped": scored["prediction"] != original_pred,
                    "same_as_original": scored["prediction"] == original_pred,
                })
            row = {
                "version": "semantic_flip_entity_swap_v1",
                "status": "ok",
                "model": model_name,
                "dataset": dataset,
                "qtype": "closed",
                "dataset_index": idx,
                "sample_id": sample_id_for(idx, sample),
                "question_id": sample.get("question_id"),
                "image_id": sample.get("image_id"),
                "question": question,
                "ground_truth": sample.get("answer"),
                "ground_truth_yn": normalize_yes_no(sample.get("answer")),
                "original": original,
                "meta_no": {
                    "question": meta_no_question,
                    **meta_no,
                    "flipped": meta_no["prediction"] != original_pred,
                    "same_as_original": meta_no["prediction"] == original_pred,
                },
                "swaps": swaps,
                "swap_count": len(swaps),
            }
        except Exception as exc:
            row = {
                "version": "semantic_flip_entity_swap_v1",
                "status": "error",
                "model": model_name,
                "dataset": dataset,
                "qtype": "closed",
                "dataset_index": idx,
                "sample_id": sample_id_for(idx, sample),
                "question_id": sample.get("question_id"),
                "question": sample.get("question"),
                "error": repr(exc),
            }
        append_jsonl(out_path, row)
        print(f"[{model_name}/{dataset}/semantic-swap] {idx + 1}/{len(ds)} {row['status']}", flush=True)
    summarize_file(out_path, Path(args.output_dir) / "summary" / f"{model_name}__{dataset}__closed.json")
    return out_path


def summarize_file(path: Path, out_json: Path) -> dict[str, Any]:
    rows = [r for r in load_jsonl(path) if r.get("status") == "ok"]
    meta = [r["meta_no"] for r in rows if isinstance(r.get("meta_no"), dict)]
    swap_items = []
    rows_with_swaps = 0
    for row in rows:
        swaps = row.get("swaps") or []
        if swaps:
            rows_with_swaps += 1
            swap_items.extend(swaps)
    original_yes = [1.0 if (r.get("original") or {}).get("prediction") == "yes" else 0.0 for r in rows]
    meta_flip = [1.0 if m.get("flipped") else 0.0 for m in meta]
    meta_nonflip = [1.0 if m.get("same_as_original") else 0.0 for m in meta]
    swap_flip = [1.0 if s.get("flipped") else 0.0 for s in swap_items]
    swap_same = [1.0 if s.get("same_as_original") else 0.0 for s in swap_items]
    term_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for s in swap_items:
        term = str(s.get("term") or "unknown")
        term_counts[term]["n"] += 1
        term_counts[term]["same"] += int(bool(s.get("same_as_original")))
        term_counts[term]["flip"] += int(bool(s.get("flipped")))
    summary = {
        "file": str(path),
        "n": len(rows),
        "original_yes_rate": float(np.mean(original_yes)) if original_yes else None,
        "meta_no": {
            "n": len(meta),
            "flip_rate": float(np.mean(meta_flip)) if meta_flip else None,
            "nonflip_rate": float(np.mean(meta_nonflip)) if meta_nonflip else None,
        },
        "entity_swap": {
            "rows_with_swaps": rows_with_swaps,
            "swap_instances": len(swap_items),
            "flip_rate": float(np.mean(swap_flip)) if swap_flip else None,
            "same_rate": float(np.mean(swap_same)) if swap_same else None,
        },
        "terms": {
            term: {
                "n": int(c["n"]),
                "same_rate": float(c["same"] / c["n"]) if c["n"] else None,
                "flip_rate": float(c["flip"] / c["n"]) if c["n"] else None,
            }
            for term, c in sorted(term_counts.items())
        },
    }
    write_json(out_json, summary)
    write_markdown(out_json.with_suffix(".md"), summary)
    return summary


def write_markdown(path: Path, summary: Mapping[str, Any]) -> None:
    meta = summary.get("meta_no") or {}
    swap = summary.get("entity_swap") or {}
    lines = [
        "# Semantic Flip and Entity-Swap Probe",
        "",
        f"- source: `{summary.get('file')}`",
        f"- n: {summary.get('n')}",
        f"- original yes rate: {fmt(summary.get('original_yes_rate'))}",
        "",
        "| probe | n | flip rate | non-flip / same rate |",
        "|---|---:|---:|---:|",
        f"| meta-no semantic flip | {meta.get('n')} | {fmt(meta.get('flip_rate'))} | {fmt(meta.get('nonflip_rate'))} |",
        f"| entity swap instances | {swap.get('swap_instances')} | {fmt(swap.get('flip_rate'))} | {fmt(swap.get('same_rate'))} |",
        "",
        f"- rows with entity swaps: {swap.get('rows_with_swaps')}",
    ]
    terms = summary.get("terms") or {}
    if terms:
        lines.extend(["", "| swapped term | n | flip rate | same rate |", "|---|---:|---:|---:|"])
        for term, row in sorted(terms.items(), key=lambda kv: -int(kv[1].get("n", 0))):
            lines.append(f"| {term} | {row.get('n')} | {fmt(row.get('flip_rate'))} | {fmt(row.get('same_rate'))} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    summaries = []
    for path in sorted((out_dir / "raw").glob("*__*__closed.jsonl")):
        parts = path.stem.split("__")
        if len(parts) != 3:
            continue
        model, dataset, _qtype = parts
        summary_path = out_dir / "summary" / f"{model}__{dataset}__closed.json"
        summary = summarize_file(path, summary_path)
        summary.update({"model": model, "dataset": dataset})
        summaries.append(summary)
    write_json(out_dir / "summary" / "all_settings.json", {"settings": summaries})
    lines = [
        "# Semantic Flip and Entity-Swap Probe: All Settings",
        "",
        "| model | dataset | n | meta-no flip | meta-no non-flip | swap instances | swap flip | swap same |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        meta = row.get("meta_no") or {}
        swap = row.get("entity_swap") or {}
        lines.append(
            f"| {row.get('model')} | {row.get('dataset')} | {row.get('n')} | "
            f"{fmt(meta.get('flip_rate'))} | {fmt(meta.get('nonflip_rate'))} | "
            f"{swap.get('swap_instances')} | {fmt(swap.get('flip_rate'))} | {fmt(swap.get('same_rate'))} |"
        )
    (out_dir / "summary").mkdir(parents=True, exist_ok=True)
    (out_dir / "summary" / "all_settings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    for model_name in args.models:
        print(f"Loading model {model_name}", flush=True)
        model, processor, model_config = medvlm.load_model(
            model_name,
            quantization=args.quantization,
            use_flash_attention=not args.no_flash_attention,
            device_map=args.device_map,
        )
        for dataset in args.datasets:
            score_dataset(args, model_name, dataset, model, processor, model_config)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    aggregate(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS))
    p.add_argument("--output-dir", default=str(DEFAULT_OUT))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-samples", type=int)
    p.add_argument("--max-swaps", type=int, default=2)
    p.add_argument("--quantization")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--no-flash-attention", action="store_true")
    p.add_argument("--resume", action="store_true")

    p = sub.add_parser("aggregate")
    p.add_argument("--output-dir", default=str(DEFAULT_OUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "run":
        run(args)
    elif args.cmd == "aggregate":
        aggregate(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
