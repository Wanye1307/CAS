#!/usr/bin/env python3
"""Run the seven-model, three-dataset closed-question CAS benchmark.

Runs sequentially on the selected device. No model or dataset is downloaded
by the dry-run mode.
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

def expected_tasks() -> list[tuple[str, str, str]]:
    return [(model, dataset, "closed") for model in MAIN_MODELS for dataset in DATASETS]


def require_inputs() -> None:
    missing = []
    for model, dataset, qtype in expected_tasks():
        name = f"{model}__{dataset}__{qtype}.jsonl"
        for rel in (
            "02_inputs/greedy_answers", "02_inputs/labels", "03_runs/hac/raw",
            "03_runs/cas_closed/features", "03_runs/ptrue/raw",
        ):
            path = ROOT / rel / name
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(path)
    if missing:
        preview = "\n".join(str(path) for path in missing[:12])
        raise SystemExit(f"Missing or empty main-experiment input files ({len(missing)}):\n{preview}")


def verify_summary_scope() -> None:
    path = ROOT / "04_results/closed/json/summary.json"
    rows = json.loads(path.read_text(encoding="utf-8"))["settings"]
    actual = {(row["model"], row["dataset"], row["qtype"]) for row in rows}
    expected = set(expected_tasks())
    if len(rows) != len(expected) or actual != expected:
        raise SystemExit(f"Unexpected closed summary scope: rows={len(rows)}, "
                         f"expected={len(expected)}, missing={sorted(expected - actual)}, "
                         f"extra={sorted(actual - expected)}")


def verify_paper_metrics_scope() -> None:
    path = ROOT / "04_results/paper_metrics/benchmark_main_metrics.json"
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    closed = sum(row["task"] == "closed" for row in rows)
    if (len(rows), closed) != (210, 210):
        raise SystemExit(f"Unexpected paper metrics scope: total={len(rows)}, "
                         f"closed={closed}; expected 210 closed rows")


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
        for model in MAIN_MODELS:
            qtype = "closed"
            base = ["--models", model, "--datasets", *DATASETS]
            runtime = ["--seed", "42", "--resume", "--no-flash-attention"]
            if args.max_samples is not None:
                runtime += ["--max-samples", str(args.max_samples)]
            call("generate_greedy_answers.py", *base, "--qtypes", qtype, *runtime)
            call("label_greedy_answers.py", *base, "--qtypes", qtype)
            call("compute_hac_vase_from_greedy.py", *base, "--qtypes", qtype,
                 *runtime, "--n-samples", "10", "--temperature", "0.7",
                 "--cluster-threshold", "0.85", "--batch-size", "5")
            call("compute_ptrue_from_greedy.py", *base, "--qtypes", qtype, *runtime)
            call("compute_cas_closed_from_greedy.py", *base, *runtime)
            for dataset in DATASETS:
                call("audit_protocol_inputs.py", "--model", model, "--dataset", dataset,
                     "--qtype", qtype, "--output",
                     str(ROOT / "05_audits/protocol" / f"{model}__{dataset}__{qtype}.json"))

    if not args.dry_run:
        require_inputs()
    call("evaluate_compare_hac_cas.py", "--hac-dir", str(ROOT / "03_runs/hac/raw"),
         "--cas-dir", str(ROOT / "03_runs/cas_closed/features"),
         "--output-dir", str(ROOT / "04_results/closed"),
         "--models", *MAIN_MODELS, "--datasets", *DATASETS,
         "--outer-splits", "5", "--splitter", "stratified", "--bootstraps", "1000")
    if not args.dry_run:
        verify_summary_scope()
    if not args.no_paper_tables:
        call("build_paper_metric_tables.py", "--outer-splits", "5", "--splitter", "stratified",
             "--bootstraps", "2000", "--table-dir", str(ROOT / "04_results/tables"),
             "--models", *MAIN_MODELS, "--qtypes", "closed")
        if not args.dry_run:
            verify_paper_metrics_scope()


if __name__ == "__main__":
    main()
