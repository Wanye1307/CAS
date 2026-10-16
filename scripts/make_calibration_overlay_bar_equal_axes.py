#!/usr/bin/env python3
"""Make equal-axis overlay bar calibration figures for HAC vs CAS."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT = BENCH_ROOT / "04_results/calibration/cas_closed/paper_style/paper_calibration_main_curve_data.csv"
DEFAULT_OUTPUT = BENCH_ROOT / "04_results/calibration/cas_closed/paper_bar_style_overlay_equal_axes"
DEFAULT_METRICS = BENCH_ROOT / "04_results/paper_metrics/closed_significance.csv"

GROUP_ORDER = ["All closed settings", "VQA-RAD", "SLAKE", "VQA-Med-2019"]
BASELINE_METHOD = "HAC-family best"
CAS_METHOD = "CAS-rescaled"
BASELINE_COLOR = "#8db7d7"
BASELINE_EDGE = "#4f95c8"
CAS_COLOR = "#1f7a4f"
CAS_EDGE = "#0f5132"
DIAGONAL_COLOR = "#b7b7b7"


def load_plot_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"group", "method", "bin_left", "bin_right", "mean_confidence", "accuracy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    df = df[df["method"].isin([BASELINE_METHOD, CAS_METHOD])].copy()
    df = df[df["bin_left"] >= 0.5].copy()
    return df


def group_rows(df: pd.DataFrame, group: str, method: str) -> pd.DataFrame:
    out = df[(df["group"] == group) & (df["method"] == method)].copy()
    return out.sort_values(["bin_left", "bin_right"])


def style_axis(ax: plt.Axes, title: str, show_ylabel: bool = True, title_size: float = 20.0) -> None:
    ax.plot([0.5, 1.0], [0.5, 1.0], color=DIAGONAL_COLOR, linestyle=(0, (4, 4)), linewidth=2.0, zorder=1)
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0.5, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.arange(0.5, 1.01, 0.1))
    ax.set_yticks(np.arange(0.5, 1.01, 0.1))
    ax.set_title(title, fontsize=title_size, weight="bold", pad=10)
    ax.set_xlabel("Predicted correctness confidence", fontsize=16)
    if show_ylabel:
        ax.set_ylabel("Empirical Accuracy", fontsize=16)
    ax.tick_params(axis="both", labelsize=13, width=1.4, length=4.5, color="#777777", pad=5)
    ax.grid(axis="y", color="#e9e9e9", linewidth=1.2)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.set_facecolor("#ffffff")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#777777")
        ax.spines[side].set_linewidth(1.4)


def draw_group(ax: plt.Axes, df: pd.DataFrame, group: str, show_ylabel: bool = True, title_size: float = 20.0) -> None:
    hac = group_rows(df, group, BASELINE_METHOD)
    cas = group_rows(df, group, CAS_METHOD)
    if hac.empty and cas.empty:
        ax.axis("off")
        return

    for rows, color, edge, alpha, zorder in [
        (hac, BASELINE_COLOR, BASELINE_EDGE, 0.78, 2),
        (cas, CAS_COLOR, CAS_EDGE, 0.42, 3),
    ]:
        if rows.empty:
            continue
        left = rows["bin_left"].to_numpy(dtype=float)
        width = rows["bin_right"].to_numpy(dtype=float) - left
        height = rows["accuracy"].to_numpy(dtype=float)
        ax.bar(left, height, width=width, align="edge", color=color, edgecolor=edge, linewidth=1.0, alpha=alpha, zorder=zorder)

    if not hac.empty:
        ax.scatter(hac["mean_confidence"], hac["accuracy"], s=30, color=BASELINE_EDGE, alpha=0.85, edgecolor="none", zorder=5)
    if not cas.empty:
        ax.scatter(cas["mean_confidence"], cas["accuracy"], s=34, color=CAS_EDGE, alpha=0.95, edgecolor="none", zorder=6)

    style_axis(ax, group, show_ylabel=show_ylabel, title_size=title_size)


def load_auroc_labels(metrics_path: Path) -> dict[str, dict[str, float]]:
    if not metrics_path.exists():
        return {}
    metrics = pd.read_csv(metrics_path)
    required = {"dataset", "hac_best_auroc", "cas_auroc"}
    if not required.issubset(metrics.columns):
        return {}
    if "model" in metrics.columns:
        metrics = metrics[metrics["model"] != "llava_next_7b_local"].copy()

    out: dict[str, dict[str, float]] = {}
    out["All closed settings"] = {
        BASELINE_METHOD: float(metrics["hac_best_auroc"].mean()),
        CAS_METHOD: float(metrics["cas_auroc"].mean()),
    }
    dataset_labels = {
        "vqa_rad": "VQA-RAD",
        "slake": "SLAKE",
        "vqa_med_2019": "VQA-Med-2019",
    }
    for dataset, group in metrics.groupby("dataset"):
        label = dataset_labels.get(str(dataset), str(dataset))
        out[label] = {
            BASELINE_METHOD: float(group["hac_best_auroc"].mean()),
            CAS_METHOD: float(group["cas_auroc"].mean()),
        }
    return out


def annotate_aurocs(ax: plt.Axes, group_aurocs: dict[str, float], fontsize: float) -> None:
    if BASELINE_METHOD in group_aurocs and np.isfinite(group_aurocs[BASELINE_METHOD]):
        ax.text(
            0.035,
            0.955,
            f"HAC AUROC={group_aurocs[BASELINE_METHOD]:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=fontsize,
            color=BASELINE_EDGE,
            weight="bold",
            zorder=10,
        )
    if CAS_METHOD in group_aurocs and np.isfinite(group_aurocs[CAS_METHOD]):
        ax.text(
            0.035,
            0.890,
            f"CAS AUROC={group_aurocs[CAS_METHOD]:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=fontsize,
            color=CAS_EDGE,
            weight="bold",
            zorder=10,
        )


def legend_handles() -> list[object]:
    return [
        Line2D([0], [0], color=DIAGONAL_COLOR, linestyle=(0, (4, 4)), linewidth=2.0, label="Perfect calibration"),
        Patch(facecolor=BASELINE_COLOR, edgecolor=BASELINE_EDGE, alpha=0.78, label="HAC-family best"),
        Patch(facecolor=CAS_COLOR, edgecolor=CAS_EDGE, alpha=0.42, label="CAS"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#111111", markersize=5.0, label="Mean confidence"),
        Line2D([0], [0], color="none", label="AUROC shown in panel"),
    ]


def save_pooled_only(df: pd.DataFrame, out_dir: Path, formats: Iterable[str], aurocs: dict[str, dict[str, float]]) -> None:
    fig, ax = plt.subplots(figsize=(3.75, 3.75), dpi=360)
    draw_group(ax, df, "All closed settings", show_ylabel=True, title_size=20.0)
    annotate_aurocs(ax, aurocs.get("All closed settings", {}), fontsize=10.0)
    ax.legend(handles=legend_handles(), loc="lower center", ncol=5, frameon=True, facecolor="#fbfbf7", edgecolor="#dddddd", framealpha=0.92, fontsize=7.4, handlelength=1.2, borderpad=0.35, labelspacing=0.35, columnspacing=0.75)
    fig.subplots_adjust(left=0.16, right=0.985, top=0.88, bottom=0.16)
    for ext in formats:
        fig.savefig(out_dir / f"paper_calibration_hac_cas_overlay_bars_pooled_only_equal_axes.{ext}")
    plt.close(fig)


def save_all_groups(df: pd.DataFrame, out_dir: Path, formats: Iterable[str], aurocs: dict[str, dict[str, float]]) -> None:
    groups = [g for g in GROUP_ORDER if g in set(df["group"])]
    fig, axes = plt.subplots(1, len(groups), figsize=(10.8, 3.25), dpi=360, sharex=True, sharey=True)
    if len(groups) == 1:
        axes = [axes]
    for idx, (ax, group) in enumerate(zip(axes, groups)):
        draw_group(ax, df, group, show_ylabel=(idx == 0), title_size=12.5)
        ax.set_xlabel("Predicted Correctness Confidence", fontsize=9.5)
        if idx == 0:
            ax.set_ylabel("Empirical Accuracy", fontsize=9.5)
        ax.tick_params(labelsize=8.5)
        annotate_aurocs(ax, aurocs.get(group, {}), fontsize=6.8)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=5, frameon=True, facecolor="#fbfbf7", edgecolor="#dddddd", framealpha=0.92, fontsize=8.8, handlelength=1.35, borderpad=0.40, columnspacing=1.0)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.84, bottom=0.26, wspace=0.18)
    for ext in formats:
        fig.savefig(out_dir / f"paper_calibration_hac_cas_overlay_bars_equal_axes.{ext}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="paper_calibration_main_curve_data.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS, help="closed_significance.csv with table-aligned HAC/CAS AUROC values")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"])
    args = parser.parse_args()

    df = load_plot_data(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aurocs = load_auroc_labels(args.metrics)
    save_pooled_only(df, args.output_dir, args.formats, aurocs)
    save_all_groups(df, args.output_dir, args.formats, aurocs)
    print(f"Wrote overlay bar calibration figures to {args.output_dir}")


if __name__ == "__main__":
    main()
