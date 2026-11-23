"""Does the train/valid/test patient-structure mismatch change our tuning decisions?

The three splits have wildly different images-per-patient:

    train  11,055 patients / 30,344 images = 2.74
    valid     228 patients /  1,422 images = 6.24
    test      136 patients /  2,218 images = 16.31

If a decision made on valid (here: how much to shrink each study toward its patient mean)
flips when the metric is weighted to look like the test set, then the mismatch matters.

Method
------
1. `runs/orig_valid.csv` holds study-level probabilities after patient shrinkage a=0.85.
   That transform is invertible per patient, so recover the pre-shrink study predictions g.
2. Re-apply a grid of shrinkage values to g.
3. Score each with three weightings:
     plain     每个 study 权重相同（官方口径）
     patient   每个**患者**权重相同（消除多检查患者的主导）
     test-like 按 test 的"每患者检查次数分布"做重要性加权
4. Compare where each weighting puts the optimum.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = r"E:\挑战赛"


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


def weighted_auc(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    """Rank-based AUC with sample weights (Mann-Whitney with weights)."""
    pos, neg = y > 0, y <= 0
    n1, n0 = w[pos].sum(), w[neg].sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(p)
    # sum over positive pairs using cumulative negative weight
    cum_neg = np.cumsum(w[order] * neg[order])
    total_neg = cum_neg[-1]
    # ties: average over equal scores
    sp = p[order]
    acc = 0.0
    i = 0
    n = len(sp)
    while i < n:
        j = i
        while j + 1 < n and sp[j + 1] == sp[i]:
            j += 1
        block = order[i:j + 1]
        neg_before = cum_neg[i - 1] if i > 0 else 0.0
        neg_in = w[block][neg[block]].sum()
        # each positive in the block beats all negatives before it + half of those tied
        acc += w[block][pos[block]].sum() * (neg_before + 0.5 * neg_in)
        i = j + 1
    return float(acc / (n1 * n0))


def main() -> None:
    va = pd.read_csv(os.path.join(ROOT, "cxr_first", "valid.csv"))
    Y = np.stack([parse_label(s) for s in va.Predict_class])
    ydf = pd.DataFrame(Y, columns=range(1, 11))
    ydf["Subject_id"] = va.Subject_id.values
    ydf["Study_id"] = va.Study_id.values
    st = ydf.groupby(["Subject_id", "Study_id"])[list(range(1, 11))].max()

    pred = pd.read_csv(os.path.join(ROOT, "runs", "orig_valid.csv"))
    wide = pred.pivot_table(index=["Subject_id", "Study_id"], columns="Predict_class",
                            values="Probability", aggfunc="first").reindex(columns=range(1, 11))
    st = st.loc[wide.index]
    A = 0.85
    S = wide.to_numpy(float)

    subj = wide.index.get_level_values(0).to_numpy()
    inv = S.copy()
    for u in np.unique(subj):
        m = subj == u
        inv[m] = (S[m] - (1 - A) * S[m].mean(axis=0)) / A

    # 三种权重
    n_per = pd.Series(subj).value_counts()               # valid: studies per patient
    w_plain = np.ones(len(subj))
    w_patient = np.array([1.0 / n_per[u] for u in subj])  # 每个患者总权重 = 1
    w_patient *= len(subj) / w_patient.sum()              # 归一化到与 plain 同总和

    # test 的每患者检查次数分布 → 重要性权重
    te = pd.read_csv(os.path.join(ROOT, "cxr_first", "test.csv"))
    n_test = te.groupby("Subject_id").Study_id.nunique()
    bins = [1, 2, 3, 5, 8, 13, 21, 10**9]
    labels = ["1", "2", "3-4", "5-7", "8-12", "13-20", "21+"]
    hv = pd.cut(n_per.reindex(n_per.index).astype(int), bins=bins, labels=labels, right=False)
    ht = pd.cut(n_test.astype(int), bins=bins, labels=labels, right=False)
    pv = hv.value_counts(normalize=True)
    pt = ht.value_counts(normalize=True)
    # 按 **study** 口径匹配：让 valid 的每患者检查次数分布看起来像 test
    v_study = hv.value_counts().reindex(labels).fillna(0)
    t_study = ht.value_counts().reindex(labels).fillna(0)
    ratio = (t_study / t_study.sum()) / (v_study / v_study.sum()).replace(0, np.nan)
    ratio = ratio.fillna(0)
    w_test = np.array([ratio.get(hv[u], 0.0) for u in subj], dtype=float)
    if w_test.sum() <= 0:
        raise SystemExit("test-like 权重全为 0，检查分箱")
    w_test = w_test * len(subj) / w_test.sum()

    print("  每患者检查次数分布（有效样本量按 Kish 公式 n_eff = (Σn)²/Σn²）")
    for name, s in (("valid", n_per), ("test", n_test)):
        kish = s.sum() ** 2 / (s ** 2).sum()
        print(f"    {name:6s} 患者 {len(s):4d}  study {s.sum():5d}  n_eff = {kish:6.1f}"
              f"  （占名义 study 的 {100*kish/s.sum():.0f}%）")
    print()
    print("  分组占比对比：")
    print("    %-8s %10s %10s" % ("次数", "valid", "test"))
    for lab in labels:
        print("    %-8s %9.1f%% %9.1f%%" % (lab, 100 * pv.get(lab, 0), 100 * pt.get(lab, 0)))

    print()
    print("  不同收缩系数 × 三种加权口径下的 study 级 AUC（10 类平均）")
    print("   %6s %10s %10s %10s" % ("shrink", "plain", "patient", "test-like"))
    grid = [1.00, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50]
    res = {}
    for a in grid:
        s = a * inv.copy()
        if a < 1.0:
            pm = pd.DataFrame(inv).groupby(subj).transform("mean").to_numpy()
            s = a * inv + (1 - a) * pm
        aucs = []
        for w in (w_plain, w_patient, w_test):
            vals = [weighted_auc(st[c].to_numpy(), s[:, c - 1], w) for c in range(1, 11)]
            aucs.append(float(np.nanmean(vals)))
        res[a] = aucs
        star = " ←" if a == max(res, key=lambda k: res[k][0]) else ""
        print("   %6.2f %10.4f %10.4f %10.4f%s" % (a, aucs[0], aucs[1], aucs[2], star))

    best = {i: max(res, key=lambda k: res[k][i]) for i in range(3)}
    print()
    print("  各口径下的最优收缩系数：")
    print("    plain     %.2f  (AUC %.4f)" % (best[0], res[best[0]][0]))
    print("    patient   %.2f  (AUC %.4f)" % (best[1], res[best[1]][1]))
    print("    test-like %.2f  (AUC %.4f)" % (best[2], res[best[2]][2]))
    same = best[0] == best[2]
    print()
    print("  结论：" + ("三种口径选出同一个系数，结构差异没有改变这项决定。"
                      if same else
                      f"最优系数从 {best[0]:.2f} 改到 {best[2]:.2f} —— 患者结构差异会改变调参结论。"))


if __name__ == "__main__":
    main()
