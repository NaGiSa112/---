"""Canonical metric: study-level macro AUC on the official valid split.

Why this exists
---------------
`src/predict.py` prints an *image-level* mean AUC (and an image-level "patient smoothed"
number). Those are diagnostics, not the competition metric. The official metric is the mean
over classes of the study-level ROC-AUC, computed on the aggregated, post-processed csv.

Verified: the study-level macro AUC of runs/orig_valid.csv is 0.8251945484, which matches
`releases/final_day_seed44_2026-11-13/reports/valid_reproduction.json` exactly.

Usage
-----
    python tools/study_metric.py runs/orig_valid.csv [more.csv ...]
    python tools/study_metric.py --json out.json runs/orig_valid.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def parse_label(s) -> np.ndarray:
    v = np.zeros(10, np.float32)
    s = str(s).strip()
    if s == "" or s.lower() == "nan":
        return v
    cs = [c.strip() for c in s.split(";") if c.strip()]
    if cs == ["10"]:
        v[9] = 1.0
        return v
    for c in cs:
        i = int(c) - 1
        if i < 9:
            v[i] = 1.0
    return v


def rank_auc(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(p)
    r = np.empty(len(p), float)
    r[order] = np.arange(1, len(p) + 1)
    pos = y > 0
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def study_labels(csv_path: str | None = None) -> pd.DataFrame:
    """Study-level labels: a study is positive for a class if any of its images is."""
    path = csv_path or os.path.join(ROOT, "cxr_first", "valid.csv")
    d = pd.read_csv(path)
    Y = np.stack([parse_label(s) for s in d.Predict_class])
    lab = pd.DataFrame(Y, columns=range(1, 11))
    lab["Subject_id"] = d.Subject_id.values
    lab["Study_id"] = d.Study_id.values
    return lab.groupby(["Subject_id", "Study_id"])[list(range(1, 11))].max()


def wide_from_csv(path: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    return d.pivot_table(index=["Subject_id", "Study_id"], columns="Predict_class",
                         values="Probability", aggfunc="first").reindex(columns=range(1, 11))


def study_auc(pred_csv: str, ref_labels: pd.DataFrame | None = None,
              study_ids: list[str] | None = None) -> dict:
    st = ref_labels if ref_labels is not None else study_labels()
    w = wide_from_csv(pred_csv)
    idx = w.index.intersection(st.index)
    if study_ids is not None:
        keep = set(study_ids)
        idx = [i for i in idx if i[1] in keep]
    w, s = w.loc[idx], st.loc[idx]
    per = [rank_auc(s[c].to_numpy(), w[c].to_numpy()) for c in range(1, 11)]
    return {"studies": len(idx), "macro_auc": float(np.nanmean(per)),
            "per_class": [None if np.isnan(v) else float(v) for v in per]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    st = study_labels()
    out = {}
    print(f"  参考标签：官方 valid，{len(st)} 个 study")
    print("  %-42s %6s %14s" % ("文件", "study", "检查级宏 AUC"))
    for p in a.csv:
        if not os.path.exists(p):
            print(f"  {p}: 缺失")
            continue
        r = study_auc(p, st)
        out[os.path.relpath(p, ROOT)] = r
        print("  %-42s %6d %14.10f" % (os.path.basename(p), r["studies"], r["macro_auc"]))
        if a.json:
            print("      逐类:", " ".join(f"{v:.4f}" if v else "—" for v in r["per_class"]))
    if a.json:
        json.dump(out, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"  [wrote] {a.json}")


if __name__ == "__main__":
    main()
