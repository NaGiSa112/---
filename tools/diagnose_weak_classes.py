"""Diagnose the weak classes: is it the model, or the labels?

Classes 1 (Atelectasis), 4 (Infiltration), 6 (Pneumonia), 8 (Consolidation) have both low
AUC and unstable heatmaps. Before spending training time, three cheap tests decide which
kind of problem it is.

  1. 有多少正样本 / AUC 的置信区间有多宽      -> 能不能测出来
  2. 类间混淆：A 的预测器给 B 的标签打分     -> 标签是否纠缠
  3. 同患者不同检查的标签一致性              -> 标签是否自相矛盾
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
NAMES = {1: "Atelectasis", 2: "Cardiomegaly", 3: "Effusion", 4: "Infiltration",
         5: "Mass/Nodule", 6: "Pneumonia", 7: "Pneumothorax", 8: "Consolidation",
         9: "Edema", 10: "No Finding"}
WEAK = [1, 4, 6, 8]


def parse_label(s) -> np.ndarray:
    y = np.zeros(10, np.float32)
    s = str(s).strip()
    if s == "" or s.lower() == "nan":
        return y
    cs = [c.strip() for c in s.split(";") if c.strip()]
    if cs == ["10"]:
        y[9] = 1.0
        return y
    for c in cs:
        i = int(c) - 1
        if i < 9:
            y[i] = 1.0
    return y


def rank_auc(y, p):
    order = np.argsort(p)
    ranks = np.empty(len(p), float)
    ranks[order] = np.arange(1, len(p) + 1)
    sp = p[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    pos = y > 0
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> None:
    va = pd.read_csv(os.path.join(ROOT, "cxr_first", "valid.csv"))
    Y = np.stack([parse_label(s) for s in va.Predict_class])
    ydf = pd.DataFrame(Y, columns=range(1, 11))
    ydf["Subject_id"] = va.Subject_id.values
    ydf["Study_id"] = va.Study_id.values

    # 同患者多研究取 max 作为患者级标签（用于一致性检查）
    study = ydf.groupby(["Subject_id", "Study_id"])[list(range(1, 11))].max().reset_index()
    subj = study.groupby("Subject_id")[list(range(1, 11))].agg(lambda s: (s.sum() > 0).astype(int))
    n_multi = (study.groupby("Subject_id").size() > 1).sum()

    print("  ① 样本量与可信区间（study 级，valid）")
    print("   %-16s %6s %8s %8s %8s" % ("类别", "正样本", "AUC", "CI下", "CI上"))
    pred = pd.read_csv(os.path.join(ROOT, "runs", "orig_valid.csv"))
    wide = pred.pivot_table(index=["Subject_id", "Study_id"], columns="Predict_class",
                            values="Probability", aggfunc="first").reindex(columns=range(1, 11))
    y = study.set_index(["Subject_id", "Study_id"])[list(range(1, 11))].loc[wide.index]
    rng = np.random.default_rng(0)
    n_boot = 400
    for c in range(1, 11):
        p = wide[c].to_numpy()
        t = y[c].to_numpy()
        a = rank_auc(t, p)
        # 患者级 bootstrap（同患者整块重采样）
        subj_ids = wide.index.get_level_values(0).to_numpy()
        uniq = np.unique(subj_ids)
        boot = []
        for _ in range(n_boot):
            pick = rng.choice(uniq, len(uniq), replace=True)
            idx = np.concatenate([np.where(subj_ids == u)[0] for u in pick])
            boot.append(rank_auc(t[idx], p[idx]))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        mark = "  ← 弱类" if c in WEAK else ""
        print("   %-16s %6d %8.4f %8.4f %8.4f  宽度 %.3f%s"
              % (f"{c} {NAMES[c]}", int((study[c] > 0).sum()), a, lo, hi, hi - lo, mark))

    print()
    print("  ② 类间混淆：用 A 类的预测值去打分 B 类的标签（行=A的预测，列=B的标签）")
    print("     数字接近 1.0 说明这两个类的标签几乎不可分 → 类别定义重叠")
    cols = WEAK + [10]
    header = "   %-16s" % "" + "".join("%10s" % f"{b}:{NAMES[b][:8]}" for b in cols)
    print(header)
    for a_cls in cols:
        line = "   %-16s" % f"{a_cls}:{NAMES[a_cls][:10]}"
        for b_cls in cols:
            v = rank_auc(y[b_cls].to_numpy(), wide[a_cls].to_numpy())
            line += "%10.3f" % v
        print(line)

    print()
    print("  ③ 标签自相矛盾的程度（同一患者多次检查，报告阳性次数分布）")
    print("   %-16s %8s %10s %10s" % ("类别", "多检查患者", "阳性率", "患者内不一致率"))
    for c in WEAK:
        m = study.groupby("Subject_id")[c].sum()
        cnt = study.groupby("Subject_id")[c].size()
        multi = cnt > 1
        s, n = m[multi], cnt[multi]
        have_both = ((s > 0) & (s < n)).sum()
        rate = have_both / max(multi.sum(), 1)
        print("   %-16s %8d %9.1f%% %11.1f%%"
              % (f"{c} {NAMES[c]}", int(multi.sum()), 100 * m[multi].gt(0).mean(), 100 * rate))


if __name__ == "__main__":
    main()
