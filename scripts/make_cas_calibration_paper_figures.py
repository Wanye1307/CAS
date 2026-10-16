#!/usr/bin/env python3
"""Paper-style CAS calibration figures.

This script intentionally uses a different visual language from the internal
bar handoff plots: reliability curves with count-scaled markers, compact panels,
and one aggregated main figure plus a small representative-setting figure.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_common import BENCH_ROOT  # noqa: E402
from build_paper_metric_tables import DATASET_LABELS, DATASET_ORDER, MODEL_LABELS, MODEL_ORDER, auc  # noqa: E402
from make_cas_calibration_plots import calibration_metrics, evaluate_setting, finite_array, setting_files  # noqa: E402

METHODS = ["MaxConfidence", "HAC-family best", "CAS-rescaled"]
COLORS = {
    "MaxConfidence": "#5F6368",
    "HAC-family best": "#4E79A7",
    "CAS-rescaled": "#59A14F",
}
MARKERS = {
    "MaxConfidence": "s",
    "HAC-family best": "o",
    "CAS-rescaled": "^",
}
LINESTYLES = {
    "MaxConfidence": (0, (2, 2)),
    "HAC-family best": (0, (5, 2)),
    "CAS-rescaled": "solid",
}


def clip_prob(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), 1e-6, 1.0 - 1e-6)


def binned_curve(correct: Sequence[float], confidence: Sequence[float], bins: int) -> pd.DataFrame:
    y = np.asarray(correct, dtype=float)
    p = clip_prob(np.asarray(confidence, dtype=float))
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for i in range(bins):
        left, right = float(edges[i]), float(edges[i + 1])
        if i == bins - 1:
            m = (p >= left) & (p <= right)
        else:
            m = (p >= left) & (p < right)
        if not np.any(m):
            continue
        rows.append({
            "bin_index": i,
            "bin_left": left,
            "bin_right": right,
            "bin_center": (left + right) / 2.0,
            "n": int(np.sum(m)),
            "mean_confidence": float(np.mean(p[m])),
            "accuracy": float(np.mean(y[m])),
        })
    return pd.DataFrame(rows)


def plot_reliability(ax, curves: dict[str, pd.DataFrame], title: str, show_ylabel: bool = True, legend: bool = False) -> None:
    ax.plot([0, 1], [0, 1], color="#B8B8B8", linestyle=(0, (4, 3)), linewidth=1.0, zorder=1)
    for method in METHODS:
        df = curves.get(method)
        if df is None or df.empty:
            continue
        sizes = 14 + 55 * np.sqrt(df["n"].to_numpy(dtype=float) / max(float(df["n"].max()), 1.0))
        ax.plot(
            df["mean_confidence"],
            df["accuracy"],
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            linewidth=1.9 if method == "CAS-rescaled" else 1.45,
            alpha=0.95,
            zorder=3,
        )
        ax.scatter(
            df["mean_confidence"],
            df["accuracy"],
            s=sizes,
            marker=MARKERS[method],
            facecolor=COLORS[method],
            edgecolor="#FFFFFF",
            linewidth=0.6,
            alpha=0.82,
            label=method,
            zorder=4,
        )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xlabel("Predicted correctness", fontsize=8.2)
    if show_ylabel:
        ax.set_ylabel("Empirical correctness", fontsize=8.2)
    ax.set_title(title, fontsize=9.2, weight="semibold", pad=5)
    ax.tick_params(labelsize=7.2, length=2.5, color="#666666")
    ax.grid(True, color="#ECECEC", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.set_facecolor("#FFFFFF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    if legend:
        ax.legend(
            frameon=True,
            facecolor="#FFFFFF",
            edgecolor="#DDDDDD",
            framealpha=0.92,
            fontsize=7.1,
            loc="lower right",
            handlelength=1.8,
            borderpad=0.35,
        )


def method_arrays(setting: dict[str, Any]) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    correct = np.asarray(setting["correct"], dtype=float)
    arrays = {
        "MaxConfidence": (correct, 1.0 - finite_array(setting["max_conf_risk"]), finite_array(setting["max_conf_risk"])),
        "HAC-family best": (correct, 1.0 - finite_array(setting["baseline_risk"]), finite_array(setting["baseline_risk"])),
        "CAS-rescaled": (correct, 1.0 - finite_array(setting["cas_rescaled_risk"]), finite_array(setting["cas_rescaled_risk"])),
    }
    return arrays


def concat_group(settings: list[dict[str, Any]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for method in METHODS:
        ys, ps = [], []
        for setting in settings:
            correct, conf, _risk = method_arrays(setting)[method]
            ys.append(correct)
            ps.append(conf)
        out[method] = (np.concatenate(ys), np.concatenate(ps))
    return out


def curves_for_settings(settings: list[dict[str, Any]], bins: int) -> dict[str, pd.DataFrame]:
    arrays = concat_group(settings)
    return {method: binned_curve(correct, conf, bins) for method, (correct, conf) in arrays.items()}


def setting_metric_rows(settings: list[dict[str, Any]], bins: int) -> list[dict[str, Any]]:
    rows = []
    for setting in settings:
        arrays = method_arrays(setting)
        for method, (correct, conf, risk) in arrays.items():
            row = {
                "model": setting["model"],
                "dataset": setting["dataset"],
                "model_label": MODEL_LABELS[setting["model"]],
                "dataset_label": DATASET_LABELS[setting["dataset"]],
                "setting": setting["task_label"],
                "method": method,
                "auroc": auc(1 - correct.astype(int), risk),
            }
            row.update(calibration_metrics(correct, conf, bins))
            rows.append(row)
    return rows


def write_curve_csv(path: Path, grouped: dict[str, dict[str, pd.DataFrame]]) -> None:
    rows = []
    for group, curves in grouped.items():
        for method, df in curves.items():
            if df.empty:
                continue
            for rec in df.to_dict("records"):
                rows.append({"group": group, "method": method, **rec})
    pd.DataFrame(rows).to_csv(path, index=False)


def choose_representatives(metric_df: pd.DataFrame) -> pd.DataFrame:
    pivot = metric_df.pivot_table(index=["model", "dataset", "setting"], columns="method", values=["ece", "brier", "auroc"], aggfunc="first")
    rows = []
    for idx, rec in pivot.iterrows():
        model, dataset, setting = idx
        rows.append({
            "model": model,
            "dataset": dataset,
            "setting": setting,
            "ece_gain_vs_hac": float(rec[("ece", "HAC-family best")] - rec[("ece", "CAS-rescaled")]),
            "brier_gain_vs_hac": float(rec[("brier", "HAC-family best")] - rec[("brier", "CAS-rescaled")]),
            "auroc_gain_vs_hac": float(rec[("auroc", "CAS-rescaled")] - rec[("auroc", "HAC-family best")]),
            "cas_ece": float(rec[("ece", "CAS-rescaled")]),
            "hac_ece": float(rec[("ece", "HAC-family best")]),
        })
    df = pd.DataFrame(rows)
    picks = []
    picks.append(("largest ECE gain", df.sort_values("ece_gain_vs_hac", ascending=False).iloc[0]))
    picks.append(("largest AUROC gain", df.sort_values("auroc_gain_vs_hac", ascending=False).iloc[0]))
    median_idx = (df["ece_gain_vs_hac"] - df["ece_gain_vs_hac"].median()).abs().sort_values().index[0]
    picks.append(("median ECE gain", df.loc[median_idx]))
    loss_df = df.sort_values("ece_gain_vs_hac", ascending=True)
    picks.append(("failure or limitation", loss_df.iloc[0]))
    out = []
    seen = set()
    for reason, row in picks:
        key = (row["model"], row["dataset"])
        if key in seen:
            continue
        seen.add(key)
        rec = row.to_dict()
        rec["selection_reason"] = reason
        out.append(rec)
    for _, row in df.sort_values("ece_gain_vs_hac", ascending=False).iterrows():
        if len(out) >= 4:
            break
        key = (row["model"], row["dataset"])
        if key in seen:
            continue
        seen.add(key)
        rec = row.to_dict()
        rec["selection_reason"] = "additional high-gain nonduplicate"
        out.append(rec)
    return pd.DataFrame(out)


def save_main_figure(settings: list[dict[str, Any]], out_dir: Path, bins: int) -> None:
    grouped_curves = {"All closed settings": curves_for_settings(settings, bins)}
    for dataset in DATASET_ORDER:
        members = [s for s in settings if s["dataset"] == dataset]
        if members:
            grouped_curves[DATASET_LABELS[dataset]] = curves_for_settings(members, bins)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 4, figsize=(10.6, 2.65), dpi=360, sharex=True, sharey=True)
    for i, (title, curves) in enumerate(grouped_curves.items()):
        plot_reliability(axes[i], curves, title, show_ylabel=(i == 0), legend=(i == 0))
    fig.subplots_adjust(left=0.055, right=0.995, top=0.87, bottom=0.20, wspace=0.18)
    fig.savefig(out_dir / "paper_calibration_main_pooled_and_dataset.png")
    fig.savefig(out_dir / "paper_calibration_main_pooled_and_dataset.pdf")
    plt.close(fig)
    write_curve_csv(out_dir / "paper_calibration_main_curve_data.csv", grouped_curves)


def save_representative_figure(settings: list[dict[str, Any]], selected: pd.DataFrame, out_dir: Path, bins: int) -> None:
    by_key = {(s["model"], s["dataset"]): s for s in settings}
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.4), dpi=360, sharex=True, sharey=True)
    axes = axes.ravel()
    for i, row in selected.reset_index(drop=True).iterrows():
        setting = by_key[(row["model"], row["dataset"])]
        curves = {method: binned_curve(*method_arrays(setting)[method][:2], bins) for method in METHODS}
        title = f"{setting['task_label']}\n{row['selection_reason']}"
        plot_reliability(axes[i], curves, title, show_ylabel=(i % 2 == 0), legend=(i == 0))
    for j in range(len(selected), len(axes)):
        axes[j].axis("off")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.10, hspace=0.34, wspace=0.16)
    fig.savefig(out_dir / "paper_calibration_representative_settings.png")
    fig.savefig(out_dir / "paper_calibration_representative_settings.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hac-dir", type=Path, default=BENCH_ROOT / "03_runs/hac/raw")
    parser.add_argument("--cas-dir", type=Path, default=BENCH_ROOT / "03_runs/cas_closed/features")
    parser.add_argument("--output-dir", type=Path, default=BENCH_ROOT / "04_results/calibration/cas_closed/paper_style")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--splitter", choices=["stratified", "kfold"], default="stratified")
    parser.add_argument("--seed", type=int, default=20260629)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    settings = []
    for model, dataset, hac_file, cas_file in setting_files(args.hac_dir, args.cas_dir):
        settings.append(evaluate_setting(model, dataset, hac_file, cas_file, args.outer_splits, args.seed, args.splitter))

    metric_df = pd.DataFrame(setting_metric_rows(settings, args.bins))
    metric_df.to_csv(args.output_dir / "paper_calibration_metrics.csv", index=False)

    selected = choose_representatives(metric_df)
    selected.to_csv(args.output_dir / "paper_calibration_selected_settings.csv", index=False)

    save_main_figure(settings, args.output_dir, args.bins)
    save_representative_figure(settings, selected, args.output_dir, args.bins)

    pooled = metric_df.groupby("method")[["ece", "mce", "brier", "nll", "auroc", "accuracy", "avg_confidence"]].mean().reset_index()
    pooled.to_csv(args.output_dir / "paper_calibration_macro_summary.csv", index=False)
    print(f"Wrote paper-style calibration figures to {args.output_dir}")


if __name__ == "__main__":
    main()
