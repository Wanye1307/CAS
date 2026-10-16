#!/usr/bin/env python3
"""Run the published 24-closed / 21-open HAC versus CAS benchmark scope.

The default scope is seven models on both question types plus LLaVA-NeXT-7B
on closed questions. Runs sequentially on the selected device. No model or
dataset is downloaded by the dry-run mode.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
DATASETS = ("vqa_rad", "slake", "vqa_med_2019")
MAIN_MODELS = (
    "qwen3vl_2b", "qwen3vl_8b", "internvl3_2b_local",
    "internvl3_8b_local", "lingshu_7b", "medgemma_4b", "medvlm_r1",
)
EXTRA_CLOSED_MODEL = "llava_next_7b_local"


def expected_tasks() -> list[tuple[str, str]]:
    return [(model, qtype) for model in MAIN_MODELS for qtype in ("closed", "open")] + [
        (EXTRA_CLOSED_MODEL, "closed")
    ]


def require_inputs() -> None:
    missing = []
    for model, qtype in expected_tasks():
        for dataset in DATASETS:
            name = f"{model}__{dataset}__{qtype}.jsonl"
            for rel in (
                "02_inputs/greedy_answers", "02_inputs/labels", "03_runs/hac/raw",
                "03_runs/cas_closed/features" if qtype == "closed" else "03_runs/cas_open/features",
                *(("03_runs/ptrue/raw",) if model != EXTRA_CLOSED_MODEL else ()),
            ):
                path = ROOT / rel / name
                if not path.is_file() or path.stat().st_size == 0:
                    missing.append(path)
    if missing:
        preview = "\n".join(str(path) for path in missing[:12])
        raise SystemExit(f"Missing or empty main-experiment input files ({len(missing)}):\n{preview}")


def verify_summary_scope(qtype: str) -> None:
    path = ROOT / "04_results" / qtype / "json/summary.json"
    rows = json.loads(path.read_text(encoding="utf-8"))["settings"]
    actual = {(row["model"], row["dataset"], row["qtype"]) for row in rows}
    expected = {(model, dataset, task) for model, task in expected_tasks()
                if task == qtype for dataset in DATASETS}
    if len(rows) != len(expected) or actual != expected:
        raise SystemExit(f"Unexpected {qtype} summary scope: rows={len(rows)}, "
                         f"expected={len(expected)}, missing={sorted(expected - actual)}, "
                         f"extra={sorted(actual - expected)}")


def verify_paper_metrics_scope() -> None:
    path = ROOT / "04_results/paper_metrics/benchmark_main_metrics.json"
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    closed = sum(row["task"] == "closed" for row in rows)
    opened = sum(row["task"] == "open" for row in rows)
    if (len(rows), closed, opened) != (429, 240, 189):
        raise SystemExit(f"Unexpected paper metrics scope: total={len(rows)}, "
                         f"closed={closed}, open={opened}; expected 429/240/189")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable, help="Python with MedVLM dependencies installed")
    parser.add_argument("--cuda-visible-devices", help="GPU index or indices passed to CUDA_VISIBLE_DEVICES")
    parser.add_argument("--max-samples", type=int, help="Pilot only; omit for the full benchmark")
    parser.add_argument("--evaluate-only", action="store_true", help="Use existing raw inputs")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    parser.add_argument("--no-paper-tables", action="store_true", help="Skip extra CSV/LaTeX metric tables")
    args = parser.parse_args()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", ""))))
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env["TOKENIZERS_PARALLELISM"] = "false"

    def call(script: str, *options: str) -> None:
        cmd = [args.python, str(SCRIPTS / script), *options]
        print(shlex.join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=ROOT, env=env, check=True)

    if not args.evaluate_only:
        for model, qtype in expected_tasks():
            base = ["--models", model, "--datasets", *DATASETS]
            runtime = ["--seed", "42", "--resume", "--no-flash-attention"]
            if args.max_samples is not None:
                runtime += ["--max-samples", str(args.max_samples)]
            call("generate_greedy_answers.py", *base, "--qtypes", qtype, *runtime)
            call("label_greedy_answers.py", *base, "--qtypes", qtype)
            call("compute_hac_vase_from_greedy.py", *base, "--qtypes", qtype,
                 *runtime, "--n-samples", "10", "--temperature", "0.7",
                 "--cluster-threshold", "0.85", "--batch-size", "5")
            if model != EXTRA_CLOSED_MODEL:
                call("compute_ptrue_from_greedy.py", *base, "--qtypes", qtype, *runtime)
            cas_script = "compute_cas_closed_from_greedy.py" if qtype == "closed" else "compute_cas_open_from_greedy.py"
            call(cas_script, *base, *runtime)
            for dataset in DATASETS:
                call("audit_protocol_inputs.py", "--model", model, "--dataset", dataset,
                     "--qtype", qtype, "--output",
                     str(ROOT / "05_audits/protocol" / f"{model}__{dataset}__{qtype}.json"))

    if not args.dry_run:
        require_inputs()
    for qtype in ("closed", "open"):
        cas_dir = ROOT / ("03_runs/cas_closed/features" if qtype == "closed" else "03_runs/cas_open/features")
        call("evaluate_compare_hac_cas.py", "--hac-dir", str(ROOT / "03_runs/hac/raw"),
             "--cas-dir", str(cas_dir), "--output-dir", str(ROOT / "04_results" / qtype),
             "--outer-splits", "5", "--splitter", "stratified", "--bootstraps", "1000",
             *(("--include-cas-full",) if qtype == "open" else ()))
        if not args.dry_run:
            verify_summary_scope(qtype)
    if not args.no_paper_tables:
        call("build_paper_metric_tables.py", "--outer-splits", "5", "--splitter", "stratified",
             "--bootstraps", "2000", "--table-dir", str(ROOT / "04_results/tables"))
        if not args.dry_run:
            verify_paper_metrics_scope()


if __name__ == "__main__":
    main()
