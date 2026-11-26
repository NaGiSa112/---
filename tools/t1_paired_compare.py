"""同一留出集上两个候选的配对比较（检查级宏 AUC）。

为什么配对：两个模型评的是**同一批检查**，抽样噪声大部分抵消，差值标准误比两个绝对
AUC 各自的区间小一个量级。逐条比较绝对 AUC 会把"没差别"读成"差不多"或"更差"。

    python tools/t1_paired_compare.py --a A_probs.csv --b B_probs.csv --split fold2 --reps 2000

输出：宏 AUC 的 A/B/Δ + 配对 bootstrap 95% 区间 + 逐类 Δ。
CSV 必须是 T2 §2.1 的逐图概率（本脚本自己做检查级聚合）。
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

PCOLS = [f"p{k}" for k in range(1, 11)]


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


def rank_auc(y, p) -> float:
    order = np.argsort(p, kind="mergesort")
    ps = p[order]
    r = np.empty(len(p), float)
    i = 0
    while i < len(ps):
        j = i
        while j + 1 < len(ps) and ps[j + 1] == ps[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    pos = y > 0
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def to_study(df: pd.DataFrame, probs: np.ndarray, lat_weight: float, shrink: float):
    d = pd.DataFrame(probs, columns=PCOLS)
    d["Subject_id"] = df["Subject_id"].values
    d["Study_id"] = df["Study_id"].values
    w = np.where(df["ViewPosition"].fillna("NA").astype(str).isin(["LATERAL", "LL"]).values,
                 lat_weight, 1.0)
    for c in PCOLS:
        d[c] = d[c] * w
    g = d.groupby(["Subject_id", "Study_id"], as_index=False)[PCOLS].sum()
    wsum = pd.Series(w).groupby([df["Subject_id"].values, df["Study_id"].values]).sum()
    g[PCOLS] = g[PCOLS].div(wsum.values[:, None])
    lab = (df.assign(k=df.Study_id.astype(str)).drop_duplicates("k")[["Study_id", "Predict_class"]])
    g = g.merge(lab, on="Study_id", how="left")
    P = g[PCOLS].values.astype(np.float64)
    if shrink < 1.0:
        pm = g.groupby("Subject_id")[PCOLS].transform("mean").values
        P = shrink * P + (1.0 - shrink) * pm
    Y = np.stack([parse_label(v) for v in g["Predict_class"].astype(str)])
    return g, Y, P


def macro(Y, P, idx=None):
    if idx is not None:
        Y, P = Y[idx], P[idx]
    return float(np.nanmean([rank_auc(Y[:, c], P[:, c]) for c in range(10)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--split", default="fold2", choices=["fold2", "valid"])
    ap.add_argument("--lat-weight", type=float, default=0.6)
    ap.add_argument("--shrink", type=float, default=0.85)
    ap.add_argument("--unit", default="patient", choices=["patient", "study"],
                    help="重采样单位；仓库 compare_folds.py 用 patient（同一患者的多次检查相关，"
                         "按检查重采样会把区间说得过窄）")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261126)
    a = ap.parse_args()

    data = os.environ.get("CXR_DATA", "")
    if a.split == "valid":
        src = pd.read_csv(os.path.join(data, "valid.csv"))
    else:
        tr = pd.read_csv(os.path.join(data, "train.csv"))
        fk = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      "data", "patient_folds_5f.csv"))
        fm = dict(zip(fk.Study_id.astype(str), fk.fold.astype(int)))
        src = tr[tr.Study_id.astype(str).map(fm) == 2].reset_index(drop=True)

    da = pd.read_csv(a.a, dtype={"Dicom_id": str})
    db = pd.read_csv(a.b, dtype={"Dicom_id": str})
    src = src.assign(_k=src.Dicom_id.astype(str)).set_index("_k")
    da = da.assign(_k=da.Dicom_id.astype(str)).set_index("_k").loc[src.index]
    db = db.assign(_k=db.Dicom_id.astype(str)).set_index("_k").loc[src.index]
    df = src.reset_index(drop=True)

    g0, Y, PA = to_study(df, da[PCOLS].values.astype(np.float64), a.lat_weight, a.shrink)
    _, Y2, PB = to_study(df, db[PCOLS].values.astype(np.float64), a.lat_weight, a.shrink)
    assert (Y == Y2).all(), "两次的检查级标签不一致"
    n = len(Y)

    ma, mb = macro(Y, PA), macro(Y, PB)
    print(f"studies {n}  A {ma:.4f}  B {mb:.4f}  delta {mb - ma:+.4f}")

    rng = np.random.default_rng(a.seed)
    if a.unit == "patient":
        # 仓库口径（compare_folds.py）：按患者有放回重采样，同一患者的检查整组进出。
        # 同一患者的多张检查高度相关，按检查重采样会把 CI 说得过窄。
        subj = g0["Subject_id"].to_numpy()
        uniq = np.unique(subj)
        rows_of = {u: np.where(subj == u)[0] for u in uniq}
    else:
        uniq = np.arange(n)
        rows_of = {u: np.array([u]) for u in uniq}

    boot = []
    for _ in range(a.reps):
        pick = rng.choice(uniq, len(uniq), replace=True)
        take = np.concatenate([rows_of[u] for u in pick])
        boot.append(macro(Y, PB, take) - macro(Y, PA, take))
    boot = np.array(boot)
    lo, hi = np.nanpercentile(boot, 2.5), np.nanpercentile(boot, 97.5)
    star = "  <-- 区间不含 0" if (lo > 0 or hi < 0) else ""
    print(f"{a.unit} 级配对 bootstrap 95%CI [{lo:+.4f}, {hi:+.4f}]  "
          f"({a.reps} 次, {len(uniq)} 个{a.unit}){star}")

    print("逐类 Δ (B - A)：")
    for c in range(10):
        ca = rank_auc(Y[:, c], PA[:, c])
        cb = rank_auc(Y[:, c], PB[:, c])
        print(f"  c{c+1}: {ca:.3f} -> {cb:.3f}  {cb - ca:+.3f}")


if __name__ == "__main__":
    main()
