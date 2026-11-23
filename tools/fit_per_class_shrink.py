"""逐类患者收缩系数：在五折 OOF 上拟合 a_c，并用留一折验证。

背景
----
现行后处理对所有类别用同一个收缩系数：
    p = a·p_study + (1-a)·p_患者均值,  a = 0.85
但不同类别对"患者先验"的依赖不同：实变/水肿在患者内高度相关（患者均值可信），
而肺不张常为单次即发（患者均值可能反而模糊）。

方法
----
1. OOF 的 study 概率已施加过 a0=0.85 的收缩，先逐患者反解出未收缩的 g。
2. 对每个类在网格上拟合 a_c；同时拟合一个全局 a 作对照。
3. 留一折（LOFO）验证：在 4 折上拟合 a_c，在该折上评估，再合并全部留出折算宏 AUC。
   这样报的是**样本外**增益，而不是在全量上拟合的乐观值。

    python tools/fit_per_class_shrink.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = r"E:\挑战赛"
PCOLS = list(range(1, 11))
NAMES = {1: "Atelectasis", 2: "Cardiomegaly", 3: "Effusion", 4: "Infiltration",
         5: "Mass/Nodule", 6: "Pneumonia", 7: "Pneumothorax", 8: "Consolidation",
         9: "Edema", 10: "No Finding"}
A0 = 0.85          # predict_fold 施加的收缩系数
GRID = np.round(np.arange(0.40, 1.001, 0.05), 2)

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


def auc(y, p) -> float:
    order = np.argsort(p)
    r = np.empty(len(p), float)
    r[order] = np.arange(1, len(p) + 1)
    pos = y > 0
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def wide(path: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    return d.pivot_table(index=["Subject_id", "Study_id"], columns="Predict_class",
                         values="Probability", aggfunc="first").reindex(columns=PCOLS)


def patient_mean(df: pd.DataFrame) -> pd.DataFrame:
    subj = df.index.get_level_values(0)
    return df.groupby(subj).transform("mean")


def shrink(df: pd.DataFrame, a: float) -> pd.DataFrame:
    pm = patient_mean(df)
    return a * df + (1 - a) * pm


def macro_auc(df: pd.DataFrame, lab: pd.DataFrame, cols=None) -> tuple[float, list]:
    cols = cols or PCOLS
    per = [auc(lab[c].to_numpy(), df[c].to_numpy()) for c in cols]
    return float(np.nanmean(per)), per


def main() -> None:
    tr = pd.read_csv(os.path.join(ROOT, "cxr_first", "train.csv"))
    Y = np.stack([parse_label(s) for s in tr.Predict_class])
    lab = pd.DataFrame(Y, columns=PCOLS)
    lab["Subject_id"] = tr.Subject_id.values
    lab["Study_id"] = tr.Study_id.values
    lab = lab.groupby(["Subject_id", "Study_id"])[PCOLS].max()

    parts = [wide(os.path.join(ROOT, "runs", f"oof_f{f}_study.csv")) for f in range(5)]
    S = pd.concat(parts)
    S = S[~S.index.duplicated()]
    idx = S.index.intersection(lab.index)
    S, lab = S.loc[idx], lab.loc[idx]
    fold = pd.read_csv(os.path.join(ROOT, "04_新一轮", "00_规则", "patient_folds_5f.csv"))
    fmap = dict(zip(fold.Study_id.astype(str), fold.fold.astype(int)))
    fof = np.array([fmap.get(str(i[1]), -1) for i in S.index])
    print(f"  OOF: {len(S)} study / {S.index.get_level_values(0).nunique()} 患者")

    # 反解出未收缩的 g
    G = (S - (1 - A0) * patient_mean(S)) / A0

    # ---------- 全量网格：每个类各自的最优 a，以及全局最优 a ----------
    print()
    print("  === 全量拟合（仅用于看规律，不作为增益估计）")
    best_a, best_auc = {}, {}
    for c in PCOLS:
        aucs = [auc(lab[c].to_numpy(), shrink(G, a)[c].to_numpy()) for a in GRID]
        j = int(np.argmax(aucs))
        best_a[c], best_auc[c] = float(GRID[j]), float(aucs[j])
    g_aucs = [macro_auc(shrink(G, a), lab)[0] for a in GRID]
    jg = int(np.argmax(g_aucs))
    print("  %-18s %10s %10s %10s" % ("类别", "最优 a", "该 a 的 AUC", "a=0.85 的 AUC"))
    base_auc = {}
    for c in PCOLS:
        base_auc[c] = auc(lab[c].to_numpy(), shrink(G, A0)[c].to_numpy())
        print("  %-18s %10.2f %10.4f %10.4f" % (f"{c} {NAMES[c]}", best_a[c], best_auc[c],
                                                base_auc[c]))
    print("  %-18s %10.2f %10.4f %10.4f" % ("全局最优 a", GRID[jg], g_aucs[jg] - 0.0,
                                            g_aucs[int(np.argmin(np.abs(GRID - A0)))]))

    # ---------- 留一折验证 ----------
    print()
    print("  === 留一折验证（LOFO，样本外）")
    preds = {"none": G.copy(), "a085": shrink(G, A0), "perclass": pd.DataFrame(index=S.index,
                                                                               columns=PCOLS,
                                                                               dtype=float)}
    chosen = []
    for f in range(5):
        fit = fof != f
        held = fof == f
        if held.sum() == 0:
            continue
        a_c = {}
        for c in PCOLS:
            aucs = [auc(lab[c].to_numpy()[fit], shrink(G, a)[c].to_numpy()[fit]) for a in GRID]
            a_c[c] = float(GRID[int(np.argmax(aucs))])
        chosen.append(a_c)
        # 用训练折拟合的 a_c 去处理留出折
        for c in PCOLS:
            col = a_c[c] * G[c] + (1 - a_c[c]) * G[c].groupby(
                G.index.get_level_values(0)).transform("mean")
            preds["perclass"].loc[held, c] = col[held].to_numpy()
        print("    fold %d: " % f + " ".join(f"{a_c[c]:.2f}" for c in PCOLS))

    rows = []
    for name, df in preds.items():
        m, per = macro_auc(df, lab)
        rows.append((name, m, per))
    print()
    print("  %-14s %10s %s" % ("配置", "宏 AUC", "相对 a=0.85"))
    ref = dict((r[0], r[1]) for r in rows)["a085"]
    for name, m, per in rows:
        tag = {"none": "不收缩 (a=1)", "a085": "现行 a=0.85", "perclass": "逐类拟合"}[name]
        print("  %-14s %10.4f %+10.4f" % (tag, m, m - ref))
    print()
    print("  %-18s %10s %10s %10s" % ("类别", "不收缩", "现行0.85", "逐类拟合"))
    for c in PCOLS:
        v = {name: per[c - 1] for name, m, per in rows}
        print("  %-18s %10.4f %10.4f %10.4f" % (f"{c} {NAMES[c]}", v["none"], v["a085"],
                                                v["perclass"]))

    out = {"a0": A0,
           "grid_best_per_class": {str(k): v for k, v in best_a.items()},
           "lofo_chosen": [{str(k): v for k, v in d.items()} for d in chosen],
           "macro": {name: m for name, m, _ in rows},
           "per_class": {name: per for name, m, per in rows}}
    p = os.path.join(ROOT, "reports", "per_class_shrink.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print()
    print(f"  [wrote] {p}")


if __name__ == "__main__":
    main()
