#!/usr/bin/env python3
"""Mechanism analyses explaining why CAS improves over HAC.

This script reuses cached closed-ended HAC and CAS feature files. It does not
run model inference. The outputs are paper-facing CSV and LaTeX tables for:
1. HAC-confident wrong cases,
2. HAC/CAS risk quadrants,
3. CAS high-risk versus low-risk counterfactual behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_common import BENCH_ROOT, load_jsonl  # noqa: E402
from build_paper_metric_tables import (  # noqa: E402
    CLOSED_CAS_ALL,
    DATASET_LABELS,
    MODEL_LABELS,
    auc,
    build_intersection,
    hac_calibration_cv,
    hac_risks,
    logreg_cv,
    row_features,
)
from make_cas_calibration_plots import setting_files  # noqa: E402

OUT_DIR = BENCH_ROOT / "04_results" / "mechanism_analysis"


def feature(row: dict[str, Any], key: str) -> float:
    try:
        return float((row.get("features") or {}).get(key))
    except Exception:
        return float("nan")


def summarize(df: pd.DataFrame, name: str) -> dict[str, Any]:
    return {
        "group": name,
        "n": int(len(df)),
        "error_rate": float(df["error"].mean()) if len(df) else np.nan,
        "mean_hac_risk": float(df["hac_risk"].mean()) if len(df) else np.nan,
        "mean_cas_risk": float(df["cas_risk"].mean()) if len(df) else np.nan,
        "mean_opp_shift": float(df["cas_opp_mean"].mean()) if len(df) else np.nan,
        "mean_target_shift": float(df["cas_pred_mean"].mean()) if len(df) else np.nan,
        "mean_cf_target_support": float(df["cas_cf_support_mean"].mean()) if len(df) else np.nan,
        "mean_same_pred": float(df["cas_same_pred_frac"].mean()) if len(df) else np.nan,
    }


def fmt(x: float) -> str:
    return "--" if x is None or not np.isfinite(float(x)) else f"{float(x):.3f}"


def write_table(path: Path, caption: str, label: str, rows: list[dict[str, Any]], cols: list[tuple[str, str]]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\small",
        r"\setlength{\tabcolsep}{3.0pt}",
        r"\begin{tabular}{@{}l" + "r" * (len(cols) - 1) + r"@{}}",
        r"\toprule",
        " & ".join(head for _, head in cols) + " " + chr(92) * 2,
        r"\midrule",
    ]
    for row in rows:
        vals = []
        for key, _ in cols:
            if key == "group":
                vals.append(str(row[key]).replace("%", r"\%"))
            elif key == "n":
                vals.append(str(int(row[key])))
            else:
                vals.append(fmt(row[key]))
        lines.append(" & ".join(vals) + " " + chr(92) * 2)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    settings: list[dict[str, Any]] = []

    for model, dataset, hac_file, cas_file in setting_files(BENCH_ROOT / "03_runs/hac/raw", BENCH_ROOT / "03_runs/cas_closed/features"):
        if model == "llava_next_7b_local":
            continue
        hac_rows, cas_rows, ids = build_intersection(load_jsonl(hac_file), load_jsonl(cas_file))
        error = np.asarray([0 if bool(r["is_correct"]) else 1 for r in hac_rows], dtype=int)
        raw_hac = hac_risks(hac_rows)
        confidence = np.asarray([float(r.get("hac_confidence_n10")) for r in hac_rows], dtype=float)
        vase = np.asarray([float(r.get("VASE")) for r in hac_rows], dtype=float)
        hac_candidates = {"HAC": np.asarray(raw_hac["HAC"], dtype=float)}
        hac_candidates["HAC-Platt"] = np.asarray(hac_calibration_cv(error, confidence, vase, "hac_platt", 5, 42, "stratified")["scores_oof"], dtype=float)
        hac_candidates["HAC-Gate"] = np.asarray(hac_calibration_cv(error, confidence, vase, "hac_gate", 5, 42, "stratified")["scores_oof"], dtype=float)
        best_name = max(hac_candidates, key=lambda name: float(auc(error, hac_candidates[name]) or -1.0))
        hac_risk = hac_candidates[best_name]
        cas_x = row_features(cas_rows, CLOSED_CAS_ALL)
        cas_risk = np.asarray(logreg_cv(error, cas_x, 5, 42, "stratified")["scores_oof"], dtype=float)

        settings.append({
            "model": model,
            "dataset": dataset,
            "n": int(len(error)),
            "n_error": int(error.sum()),
            "hac_best": best_name,
            "hac_auc": auc(error, hac_risk),
            "cas_auc": auc(error, cas_risk),
        })
        for i, (hrow, crow, sid) in enumerate(zip(hac_rows, cas_rows, ids)):
            rows.append({
                "model": model,
                "dataset": dataset,
                "model_label": MODEL_LABELS[model],
                "dataset_label": DATASET_LABELS[dataset],
                "sample_id": sid,
                "dataset_index": crow.get("dataset_index"),
                "question": crow.get("question"),
                "ground_truth": crow.get("ground_truth"),
                "target_prediction": crow.get("target_prediction"),
                "is_correct": bool(crow.get("is_correct")),
                "error": int(error[i]),
                "hac_best": best_name,
                "hac_risk": float(hac_risk[i]),
                "cas_risk": float(cas_risk[i]),
                "cas_pred_mean": feature(crow, "cas_pred_mean"),
                "cas_pred_max": feature(crow, "cas_pred_max"),
                "cas_opp_mean": feature(crow, "cas_opp_mean"),
                "cas_opp_max": feature(crow, "cas_opp_max"),
                "cas_cf_support_mean": feature(crow, "cas_cf_support_mean"),
                "cas_cf_support_max": feature(crow, "cas_cf_support_max"),
                "cas_same_pred_frac": feature(crow, "cas_same_pred_frac"),
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "per_sample_mechanism_scores.csv", index=False)
    pd.DataFrame(settings).to_csv(OUT_DIR / "setting_mechanism_summary.csv", index=False)

    for _, idx in df.groupby(["model", "dataset"]).groups.items():
        sub = df.loc[idx]
        df.loc[idx, "hac_conf20"] = sub["hac_risk"] <= np.nanquantile(sub["hac_risk"], 0.20)
        df.loc[idx, "cas_low20"] = sub["cas_risk"] <= np.nanquantile(sub["cas_risk"], 0.20)
        df.loc[idx, "cas_high20"] = sub["cas_risk"] >= np.nanquantile(sub["cas_risk"], 0.80)
        df.loc[idx, "hac_low50"] = sub["hac_risk"] <= np.nanmedian(sub["hac_risk"])
        df.loc[idx, "hac_high50"] = sub["hac_risk"] > np.nanmedian(sub["hac_risk"])
        df.loc[idx, "cas_low50"] = sub["cas_risk"] <= np.nanmedian(sub["cas_risk"])
        df.loc[idx, "cas_high50"] = sub["cas_risk"] > np.nanmedian(sub["cas_risk"])

    hac_conf = df[df.hac_conf20]
    hac_summary = [
        summarize(df, "All samples"),
        summarize(hac_conf, "HAC most-confident 20%"),
        summarize(hac_conf[hac_conf.error == 0], "HAC most-confident correct"),
        summarize(hac_conf[hac_conf.error == 1], "HAC most-confident wrong"),
    ]
    quad_summary = [
        summarize(df[df.hac_low50 & df.cas_low50], "HAC low / CAS low"),
        summarize(df[df.hac_low50 & df.cas_high50], "HAC low / CAS high"),
        summarize(df[df.hac_high50 & df.cas_low50], "HAC high / CAS low"),
        summarize(df[df.hac_high50 & df.cas_high50], "HAC high / CAS high"),
    ]
    cas_summary = [
        summarize(df[df.cas_low20], "CAS lowest-risk 20%"),
        summarize(df[df.cas_high20], "CAS highest-risk 20%"),
        summarize(df[df.error == 0], "Correct samples"),
        summarize(df[df.error == 1], "Wrong samples"),
    ]

    pd.DataFrame(hac_summary).to_csv(OUT_DIR / "hac_confident_wrong_analysis.csv", index=False)
    pd.DataFrame(quad_summary).to_csv(OUT_DIR / "hac_cas_quadrant_analysis.csv", index=False)
    pd.DataFrame(cas_summary).to_csv(OUT_DIR / "cas_risk_behavior_analysis.csv", index=False)

    write_table(
        OUT_DIR / "table_hac_cas_quadrants.tex",
        "Error rates after splitting samples by within-setting median HAC and CAS risk. The HAC-low/CAS-high cell isolates cases trusted by HAC but warned by CAS.",
        "tab:hac_cas_quadrants",
        quad_summary,
        [("group", "Risk quadrant"), ("n", "$n$"), ("error_rate", "Err."), ("mean_hac_risk", "HAC"), ("mean_cas_risk", "CAS"), ("mean_opp_shift", "Opp. shift")],
    )
    write_table(
        OUT_DIR / "table_cas_risk_behavior.tex",
        "Counterfactual behavior of CAS low-risk and high-risk samples. Groups are the lowest and highest 20\\% CAS-risk samples within each model-dataset setting.",
        "tab:cas_risk_behavior",
        cas_summary[:2],
        [("group", "Subset"), ("n", "$n$"), ("error_rate", "Err."), ("mean_cf_target_support", "CF support"), ("mean_opp_shift", "Opp. shift"), ("mean_same_pred", "Same pred.")],
    )
    write_table(
        OUT_DIR / "table_hac_confident_wrong.tex",
        "CAS re-ranks samples in the HAC-confident region. HAC most-confident samples are the lowest-risk 20\\% within each model-dataset setting.",
        "tab:hac_confident_wrong",
        hac_summary[1:],
        [("group", "Subset"), ("n", "$n$"), ("error_rate", "Err."), ("mean_hac_risk", "HAC risk"), ("mean_cas_risk", "CAS risk"), ("mean_opp_shift", "Opp. shift")],
    )

    print(f"Wrote mechanism analysis to {OUT_DIR}")
    print(pd.DataFrame(hac_summary).to_string(index=False))
    print(pd.DataFrame(quad_summary).to_string(index=False))
    print(pd.DataFrame(cas_summary).to_string(index=False))


if __name__ == "__main__":
    main()
