#!/usr/bin/env python3
"""Build CAS closed-ended calibration tables and reliability plots.

This mirrors the example-style reliability-bar workflow used in
YekaiShen/T1UQ, but uses the greedy HAC/CAS benchmark caches. The baseline is
selected per setting from the HAC family (HAC, HAC-Platt, HAC-Gate) using the
same OOF AUROC protocol as the paper tables; CAS uses the closed CAS-only OOF
logistic score. Scores are converted to predicted correctness confidence.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from sklearn.linear_model import LogisticRegression

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_common import BENCH_ROOT, load_jsonl, output_path, row_sample_id, write_json  # noqa: E402
from build_paper_metric_tables import (  # noqa: E402
    CLOSED_CAS_ALL,
    DATASET_LABELS,
    DATASET_ORDER,
    MODEL_LABELS,
    MODEL_ORDER,
    auc,
    build_intersection,
    hac_calibration_cv,
    hac_risks,
    logreg_cv,
    raw_closed_risks,
    row_features,
    split_indices,
    standardize,
)

BASELINE_COLOR = "#5DADD5"
CAS_COLOR = "#E67AA8"
EDGE = "#222222"


def finite_array(x: Sequence[float]) -> np.ndarray:
    return np.asarray(x, dtype=float)


def clip_prob(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), 1e-6, 1.0 - 1e-6)


def calibration_metrics(correct: np.ndarray, confidence: np.ndarray, n_bins: int) -> dict[str, Any]:
    correct = np.asarray(correct, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    mask = np.isfinite(correct) & np.isfinite(confidence)
    y = correct[mask]
    p = clip_prob(confidence[mask])
    if len(y) == 0:
        return {"n_valid": 0, "ece": None, "mce": None, "brier": None, "nll": None, "avg_confidence": None, "accuracy": None}
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            in_bin = (p >= edges[i]) & (p <= edges[i + 1])
        else:
            in_bin = (p >= edges[i]) & (p < edges[i + 1])
        if not np.any(in_bin):
            continue
        gap = abs(float(np.mean(y[in_bin])) - float(np.mean(p[in_bin])))
        ece += float(np.mean(in_bin)) * gap
        mce = max(mce, gap)
    return {
        "n_valid": int(len(y)),
        "ece": float(ece),
        "mce": float(mce),
        "brier": float(np.mean((p - y) ** 2)),
        "nll": float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))),
        "avg_confidence": float(np.mean(p)),
        "accuracy": float(np.mean(y)),
    }


def logit_prob(x: np.ndarray) -> np.ndarray:
    x = clip_prob(np.asarray(x, dtype=float))
    return np.log(x / (1.0 - x))


def cas_rescaled_cv(labels: np.ndarray, x: np.ndarray, n_splits: int, seed: int, splitter: str) -> dict[str, Any]:
    """Return OOF CAS error risks before and after fold-local probability rescaling.

    The base CAS detector keeps the paper protocol: class-balanced logistic
    regression for error ranking. Its probability scale is not calibrated
    because class balancing changes the effective class prior. For calibration
    plots we fit a one-dimensional Platt rescaler on the training fold's CAS
    probabilities and apply it to that fold's held-out scores.
    """
    labels = np.asarray(labels, dtype=int)
    raw_oof = np.full(len(labels), np.nan, dtype=float)
    rescaled_oof = np.full(len(labels), np.nan, dtype=float)
    folds = []
    if len(labels) < n_splits or len(set(labels.tolist())) < 2:
        return {"raw_risk_oof": raw_oof.tolist(), "rescaled_risk_oof": rescaled_oof.tolist(), "folds": folds}

    for fold, (train_idx, test_idx) in enumerate(split_indices(labels, n_splits, seed, splitter)):
        if len(set(labels[train_idx].tolist())) < 2 or len(set(labels[test_idx].tolist())) < 2:
            continue
        x_train, x_test = standardize(x[train_idx], x[test_idx])
        clf = LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=seed)
        clf.fit(x_train, labels[train_idx])
        train_raw = clf.predict_proba(x_train)[:, 1]
        test_raw = clf.predict_proba(x_test)[:, 1]
        raw_oof[test_idx] = test_raw

        cal = LogisticRegression(C=1.0, class_weight=None, solver="liblinear", max_iter=1000, random_state=seed)
        cal.fit(logit_prob(train_raw).reshape(-1, 1), labels[train_idx])
        test_rescaled = cal.predict_proba(logit_prob(test_raw).reshape(-1, 1))[:, 1]
        rescaled_oof[test_idx] = test_rescaled
        folds.append({"fold": int(fold), "train_n": int(len(train_idx)), "test_n": int(len(test_idx))})

    return {"raw_risk_oof": raw_oof.tolist(), "rescaled_risk_oof": rescaled_oof.tolist(), "folds": folds}


def bin_rows(task: str, correct: np.ndarray, baseline_conf: np.ndarray, cas_conf: np.ndarray, n_bins: int) -> list[dict[str, Any]]:
    correct = np.asarray(correct, dtype=float)
    baseline_conf = np.asarray(baseline_conf, dtype=float)
    cas_conf = np.asarray(cas_conf, dtype=float)
    mask = np.isfinite(correct) & np.isfinite(baseline_conf) & np.isfinite(cas_conf)
    y = correct[mask]
    b = clip_prob(baseline_conf[mask])
    c = clip_prob(cas_conf[mask])
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, Any]] = []
    for i in range(n_bins):
        left = float(edges[i])
        right = float(edges[i + 1])
        if i == n_bins - 1:
            in_b = (b >= left) & (b <= right)
            in_c = (c >= left) & (c <= right)
        else:
            in_b = (b >= left) & (b < right)
            in_c = (c >= left) & (c < right)
        rows.append({
            "task": task,
            "bin_index": i,
            "confidence_bin": f"[{left:.1f}, {right:.1f}]" if i < n_bins - 1 else f"[{left:.1f}, {right:.1f}]",
            "bin_left": left,
            "bin_right": right,
            "bin_center": (left + right) / 2.0,
            "baseline_n": int(np.sum(in_b)),
            "baseline_accuracy": float(np.mean(y[in_b])) if np.any(in_b) else np.nan,
            "baseline_confidence": float(np.mean(b[in_b])) if np.any(in_b) else np.nan,
            "cas_n": int(np.sum(in_c)),
            "cas_accuracy": float(np.mean(y[in_c])) if np.any(in_c) else np.nan,
            "cas_confidence": float(np.mean(c[in_c])) if np.any(in_c) else np.nan,
            # Compatibility with the imported/example plotting data shape.
            "ensemble_n": int(np.sum(in_b)),
            "ensemble_accuracy": float(np.mean(y[in_b])) if np.any(in_b) else np.nan,
            "ours_n": int(np.sum(in_c)),
            "ours_accuracy": float(np.mean(y[in_c])) if np.any(in_c) else np.nan,
        })
    return rows


def setting_files(hac_dir: Path, cas_dir: Path) -> list[tuple[str, str, Path, Path]]:
    out = []
    for hac_file in sorted(hac_dir.glob("*__closed.jsonl")):
        parts = hac_file.stem.split("__")
        if len(parts) != 3:
            continue
        model, dataset, qtype = parts
        if model not in MODEL_LABELS or dataset not in DATASET_LABELS:
            continue
        cas_file = output_path(cas_dir, model, dataset, qtype)
        if cas_file.exists():
            out.append((model, dataset, hac_file, cas_file))
    out.sort(key=lambda t: (MODEL_ORDER.index(t[0]) if t[0] in MODEL_ORDER else 99, DATASET_ORDER.index(t[1]) if t[1] in DATASET_ORDER else 99))
    return out


def evaluate_setting(model: str, dataset: str, hac_file: Path, cas_file: Path, n_splits: int, seed: int, splitter: str) -> dict[str, Any]:
    hac_rows, cas_rows, _ = build_intersection(load_jsonl(hac_file), load_jsonl(cas_file))
    error = np.asarray([0 if bool(row["is_correct"]) else 1 for row in hac_rows], dtype=int)
    correct = (error == 0).astype(float)

    raw_hac = hac_risks(hac_rows)
    confidence = np.asarray([float(row.get("hac_confidence_n10")) for row in hac_rows], dtype=float)
    vase = np.asarray([float(row.get("VASE")) for row in hac_rows], dtype=float)
    hac_candidates = {
        "HAC": np.asarray(raw_hac["HAC"], dtype=float),
    }
    hac_candidates["HAC-Platt"] = np.asarray(hac_calibration_cv(error, confidence, vase, "hac_platt", n_splits, seed, splitter)["scores_oof"], dtype=float)
    hac_candidates["HAC-Gate"] = np.asarray(hac_calibration_cv(error, confidence, vase, "hac_gate", n_splits, seed, splitter)["scores_oof"], dtype=float)
    best_name = max(hac_candidates, key=lambda name: float(auc(error, hac_candidates[name]) or -1.0))
    best_hac_risk = hac_candidates[best_name]

    cas_x = row_features(cas_rows, CLOSED_CAS_ALL)
    cas_rep = logreg_cv(error, cas_x, n_splits, seed, splitter)
    cas_risk = np.asarray(cas_rep["scores_oof"], dtype=float)
    cas_cal = cas_rescaled_cv(error, cas_x, n_splits, seed, splitter)
    cas_rescaled_risk = np.asarray(cas_cal["rescaled_risk_oof"], dtype=float)

    max_conf_risk = np.asarray(raw_closed_risks(cas_rows)["MaxConfidence"], dtype=float)

    return {
        "model": model,
        "dataset": dataset,
        "task_label": f"{MODEL_LABELS[model]} / {DATASET_LABELS[dataset]}",
        "n": int(len(error)),
        "correct": correct,
        "hac_best_name": best_name,
        "baseline_risk": best_hac_risk,
        "cas_risk": cas_risk,
        "cas_rescaled_risk": cas_rescaled_risk,
        "max_conf_risk": max_conf_risk,
    }


def style_axis(ax, title: str, ylabel: bool = True) -> None:
    ax.plot([0, 1], [0, 1], linestyle="--", color="#2B2B2B", linewidth=1.2, zorder=5)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.04)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=7.5, weight="bold", pad=4)
    ax.set_xlabel("Predicted correctness confidence", fontsize=6.5, labelpad=2)
    if ylabel:
        ax.set_ylabel("Empirical accuracy", fontsize=6.5)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.tick_params(labelsize=5.8, length=2)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_overlay(ax, task_df: pd.DataFrame, title: str, ylabel: bool = True) -> None:
    task_df = task_df.sort_values("bin_index")
    bin_width = float(task_df["bin_right"].iloc[0] - task_df["bin_left"].iloc[0])
    width = bin_width * 0.94
    left = task_df["bin_left"].to_numpy(dtype=float) + bin_width * 0.03
    base = task_df["baseline_accuracy"].to_numpy(dtype=float)
    cas = task_df["cas_accuracy"].to_numpy(dtype=float)
    valid_base = np.isfinite(base)
    valid_cas = np.isfinite(cas)
    ax.bar(left[valid_base], base[valid_base], width=width, align="edge", color=BASELINE_COLOR, edgecolor=BASELINE_COLOR, linewidth=0, alpha=0.50, zorder=2, label="HAC-family best")
    ax.bar(left[valid_cas], cas[valid_cas], width=width, align="edge", color=CAS_COLOR, edgecolor=CAS_COLOR, linewidth=0, alpha=0.48, zorder=3, label="CAS")
    style_axis(ax, title, ylabel=ylabel)


def legend_handles() -> list[Any]:
    return [
        Line2D([0], [0], color="#2B2B2B", linestyle="--", linewidth=1.2, label="Perfect calibration"),
        Patch(facecolor=BASELINE_COLOR, edgecolor=BASELINE_COLOR, alpha=0.50, label="HAC-family best"),
        Patch(facecolor=CAS_COLOR, edgecolor=CAS_COLOR, alpha=0.48, label="CAS"),
    ]


def save_plots(plot_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = list(dict.fromkeys(plot_df["task"].tolist()))
    for task in tasks:
        fig = plt.figure(figsize=(3.25, 3.0), dpi=360)
        ax = fig.add_axes([0.14, 0.14, 0.65, 0.65])
        draw_overlay(ax, plot_df[plot_df["task"] == task], task, ylabel=True)
        fig.legend(handles=legend_handles(), frameon=False, loc="center left", bbox_to_anchor=(0.79, 0.55), fontsize=5.2, handlelength=1.15, borderaxespad=0.0)
        slug = task.replace(" / ", "__").replace(" ", "_").replace("/", "-")
        fig.savefig(out_dir / f"cas_calibration_{slug}.png")
        plt.close(fig)

    ncols = 3
    nrows = int(math.ceil(len(tasks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.8, 2.55 * nrows), dpi=360, squeeze=False)
    for i, task in enumerate(tasks):
        r, c = divmod(i, ncols)
        draw_overlay(axes[r][c], plot_df[plot_df["task"] == task], task, ylabel=(c == 0))
    for j in range(len(tasks), nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")
    fig.legend(handles=legend_handles(), frameon=False, loc="center left", bbox_to_anchor=(0.86, 0.62), ncol=1, fontsize=8, handlelength=1.2, borderaxespad=0.0)
    fig.subplots_adjust(left=0.055, right=0.84, top=0.97, bottom=0.04, hspace=0.42, wspace=0.24)
    fig.savefig(out_dir / "cas_calibration_24settings_grid.png")
    plt.close(fig)


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["model", "dataset", "n", "hac_best", "method", "n_valid", "accuracy", "avg_confidence", "ece", "mce", "brier", "nll", "auroc"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hac-dir", type=Path, default=BENCH_ROOT / "03_runs/hac/raw")
    parser.add_argument("--cas-dir", type=Path, default=BENCH_ROOT / "03_runs/cas_closed/features")
    parser.add_argument("--output-dir", type=Path, default=BENCH_ROOT / "04_results/calibration/cas_closed")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--splitter", choices=["stratified", "kfold"], default="stratified")
    parser.add_argument("--seed", type=int, default=20260629)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_bins: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    aggregate_correct: list[np.ndarray] = []
    aggregate_baseline: list[np.ndarray] = []
    aggregate_cas: list[np.ndarray] = []
    aggregate_cas_rescaled: list[np.ndarray] = []

    for model, dataset, hac_file, cas_file in setting_files(args.hac_dir, args.cas_dir):
        setting = evaluate_setting(model, dataset, hac_file, cas_file, args.outer_splits, args.seed, args.splitter)
        correct = setting["correct"]
        baseline_conf = 1.0 - finite_array(setting["baseline_risk"])
        cas_conf = 1.0 - finite_array(setting["cas_risk"])
        cas_rescaled_conf = 1.0 - finite_array(setting["cas_rescaled_risk"])
        max_conf = 1.0 - finite_array(setting["max_conf_risk"])
        task = setting["task_label"]
        all_bins.extend(bin_rows(task, correct, baseline_conf, cas_rescaled_conf, args.bins))
        aggregate_correct.append(correct)
        aggregate_baseline.append(baseline_conf)
        aggregate_cas.append(cas_conf)
        aggregate_cas_rescaled.append(cas_rescaled_conf)
        for method, conf, risk in [
            ("MaxConfidence", max_conf, setting["max_conf_risk"]),
            ("HAC-family best", baseline_conf, setting["baseline_risk"]),
            ("CAS", cas_conf, setting["cas_risk"]),
            ("CAS-rescaled", cas_rescaled_conf, setting["cas_rescaled_risk"]),
        ]:
            row = {
                "model": model,
                "dataset": dataset,
                "n": setting["n"],
                "hac_best": setting["hac_best_name"],
                "method": method,
                "auroc": auc(1 - correct.astype(int), risk),
            }
            row.update(calibration_metrics(correct, conf, args.bins))
            metric_rows.append(row)

    plot_df = pd.DataFrame(all_bins)
    plot_df.to_csv(args.output_dir / "cas_calibration_plot_data_wide.csv", index=False)
    plot_df.to_csv(args.output_dir / "reliability_bar_plot_data_wide.csv", index=False)
    write_summary(args.output_dir / "cas_calibration_metrics.csv", metric_rows)

    if aggregate_correct:
        correct_all = np.concatenate(aggregate_correct)
        baseline_all = np.concatenate(aggregate_baseline)
        cas_all = np.concatenate(aggregate_cas)
        cas_rescaled_all = np.concatenate(aggregate_cas_rescaled)
        pooled_rows = []
        for method, conf in [("HAC-family best", baseline_all), ("CAS", cas_all), ("CAS-rescaled", cas_rescaled_all)]:
            row = {"method": method}
            row.update(calibration_metrics(correct_all, conf, args.bins))
            pooled_rows.append(row)
        pd.DataFrame(pooled_rows).to_csv(args.output_dir / "cas_calibration_pooled_summary.csv", index=False)
        write_json(args.output_dir / "cas_calibration_manifest.json", {
            "source": "greedy HAC/CAS closed benchmark",
            "baseline": "Per-setting HAC-family best among HAC/HAC-Platt/HAC-Gate by OOF AUROC",
            "cas_score": "CAS-only 5-fold OOF logistic error risk converted to correctness confidence",
            "cas_rescaled_score": "Fold-local one-dimensional Platt rescale fitted on training-fold CAS probabilities and applied to held-out fold",
            "plot_cas_variant": "CAS-rescaled",
            "n_settings": int(len(aggregate_correct)),
            "n_samples": int(len(correct_all)),
            "bins": int(args.bins),
            "outer_splits": int(args.outer_splits),
            "splitter": args.splitter,
            "outputs": [
                "cas_calibration_plot_data_wide.csv",
                "cas_calibration_metrics.csv",
                "cas_calibration_pooled_summary.csv",
                "cas_calibration_24settings_grid.png",
                "Note: reliability plots compare HAC-family best against CAS-rescaled; metrics CSV also contains raw CAS.",
            ],
        })

    save_plots(plot_df, args.output_dir / "example_style_hac_baseline")
    print(f"Wrote CAS calibration artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
