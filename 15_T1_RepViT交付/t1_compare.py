"""Paired comparison of two fold-2 predictions: single-model AUC, per-class Spearman,
half-power fusion delta, and a paired bootstrap CI for that delta.

Why the bootstrap: single-split AUC differences of 0.001 are inside the noise band that
this project already measured (same-recipe retrain spread 0.0014-0.0040). A delta without
an interval invites reading noise as signal.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, __file__.rsplit("07_日志与临时", 1)[0] + "14_国赛材料包/extracted")
from src.cxr.cache import label_matrix, read_split  # noqa: E402

PCOLS = [f"p{k}" for k in range(1, 11)]


def study_frame(path: str, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate image rows to studies; labels come from train.csv, not from the CSV."""
    pred = pd.read_csv(path)
    if len(pred) != len(df):
        raise SystemExit(f"[error] {path}: {len(pred)} rows vs {len(df)} images")
    if set(pred.Dicom_id) != set(df.Dicom_id):
        raise SystemExit(f"[error] {path}: Dicom_id set differs from train.csv")
    order = df.set_index("Dicom_id").loc[pred.Dicom_id].reset_index()
    y_img = label_matrix(order)
    p_img = pred[PCOLS].to_numpy(dtype=np.float64)

    g = pd.DataFrame(p_img).groupby(order.Study_id.values)
    p_st = g.mean().to_numpy()
    y_st = pd.DataFrame(y_img).groupby(order.Study_id.values).max().to_numpy()
    studies = np.array(list(g.groups.keys()))
    return studies, y_st, p_st


def macro_auc(y: np.ndarray, p: np.ndarray) -> tuple[float, np.ndarray]:
    aucs = np.array([roc_auc_score(y[:, k], p[:, k]) if y[:, k].sum() > 0 else np.nan
                     for k in range(y.shape[1])])
    return float(np.nanmean(aucs)), aucs


def spearman(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rank correlation per class over studies; robust to the two models' probability scales."""
    out = []
    for k in range(a.shape[1]):
        ra = pd.Series(a[:, k]).rank().to_numpy()
        rb = pd.Series(b[:, k]).rank().to_numpy()
        out.append(float(np.corrcoef(ra, rb)[0, 1]))
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="参考成员 fold2_probs.csv（Tiny）")
    ap.add_argument("--new", required=True, help="新成员 fold2_probs.csv（RepViT）")
    ap.add_argument("--fold-csv", default="",
                    help="折表 CSV；给了就只在该折的验证集上比较（本机只有 fold2 概率）")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    df = read_split("train")
    if a.fold_csv:
        folds = pd.read_csv(a.fold_csv, encoding="utf-8-sig")
        hold = set(folds.loc[folds.fold == 2, "Study_id"])
        df = df[df.Study_id.isin(hold)].reset_index(drop=True)
        print(f"[cmp] restricted to fold 2: {len(df)} images / {df.Study_id.nunique()} studies")
    s_base, y, p_base = study_frame(a.base, df)
    s_new, y2, p_new = study_frame(a.new, df)
    if not np.array_equal(s_base, s_new):
        raise SystemExit("[error] the two files cover different studies")
    n = len(s_base)

    auc_base, cls_base = macro_auc(y, p_base)
    auc_new, cls_new = macro_auc(y, p_new)
    # Half-power fusion: soften each member's probabilities, average, then restore scale.
    p_fus = ((np.sqrt(p_base) + np.sqrt(p_new)) / 2.0) ** 2
    auc_fus, cls_fus = macro_auc(y, p_fus)
    rho = spearman(p_base, p_new)

    rng = np.random.default_rng(a.seed)
    d_fus, d_new = [], []
    for _ in range(a.boot):
        idx = rng.integers(0, n, n)
        try:
            b0, _ = macro_auc(y[idx], p_base[idx])
            b1, _ = macro_auc(y[idx], p_fus[idx])
            b2, _ = macro_auc(y[idx], p_new[idx])
        except ValueError:
            continue
        d_fus.append(b1 - b0)
        d_new.append(b2 - b0)
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    ci_fus, ci_new = q(d_fus), q(d_new)

    res = {
        "studies": n,
        "images": len(df),
        "auc_base": auc_base,
        "auc_new": auc_new,
        "auc_fused": auc_fus,
        "delta_new_minus_base": auc_new - auc_base,
        "delta_fused_minus_base": auc_fus - auc_base,
        "delta_new_ci95": ci_new,
        "delta_fused_ci95": ci_fus,
        "spearman_per_class": [round(float(v), 4) for v in rho],
        "spearman_mean": float(rho.mean()),
        "per_class": [
            {"class": k + 1, "base": round(float(cls_base[k]), 4),
             "new": round(float(cls_new[k]), 4), "fused": round(float(cls_fus[k]), 4),
             "rho": round(float(rho[k]), 4)}
            for k in range(10)
        ],
        "criterion": {
            "delta_gt_0.003": bool((auc_fus - auc_base) > 0.003),
            "ci_low_gt_0": bool(ci_fus[0] > 0),
            "spearman_mean_lt_0.95": bool(rho.mean() < 0.95),
        },
    }
    res["criterion"]["passes"] = bool(
        res["criterion"]["delta_gt_0.003"] and res["criterion"]["ci_low_gt_0"])

    print(f"[cmp] studies={n}  base={auc_base:.6f}  new={auc_new:.6f}  fused={auc_fus:.6f}")
    print(f"[cmp] Δnew  {auc_new - auc_base:+.6f}  CI95 [{ci_new[0]:+.6f}, {ci_new[1]:+.6f}]")
    print(f"[cmp] Δfused {auc_fus - auc_base:+.6f}  CI95 [{ci_fus[0]:+.6f}, {ci_fus[1]:+.6f}]")
    print(f"[cmp] Spearman mean {rho.mean():.4f}  min {rho.min():.4f}  max {rho.max():.4f}")
    print("[cmp] per-class base/new/fused/rho")
    for row in res["per_class"]:
        print("   %2d  %.4f  %.4f  %.4f  %.3f" % (row["class"], row["base"], row["new"],
                                                   row["fused"], row["rho"]))
    print("[cmp] criterion:", json.dumps(res["criterion"], ensure_ascii=False))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(f"[cmp] wrote {a.out}")


if __name__ == "__main__":
    main()
