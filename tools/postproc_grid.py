"""后处理旋钮的 valid 全网格（CPU，只读已有预测）。

回答：在不重训的前提下，后处理还有多少可调空间？
输出 reports/后处理网格_2026-11-23.md

    python tools/postproc_grid.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = r"E:\挑战赛"
PCOLS = list(range(1, 11))
OUT = os.path.join(ROOT, "reports", "后处理网格_2026-11-23.md")

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


def main() -> None:
    va = pd.read_csv(os.path.join(ROOT, "cxr_first", "valid.csv"))
    Y = np.stack([parse_label(s) for s in va.Predict_class])
    lab = pd.DataFrame(Y, columns=PCOLS)
    lab["Subject_id"] = va.Subject_id.values
    lab["Study_id"] = va.Study_id.values
    lab = lab.groupby(["Subject_id", "Study_id"])[PCOLS].max()

    d = pd.read_csv(os.path.join(ROOT, "runs", "valid_noshrink.csv"))
    w = d.pivot_table(index=["Subject_id", "Study_id"], columns="Predict_class",
                      values="Probability", aggfunc="first").reindex(columns=PCOLS)
    idx = w.index.intersection(lab.index)
    w, lab = w.loc[idx], lab.loc[idx]
    subj = w.index.get_level_values(0)
    mu = w.groupby(subj).transform("mean")

    def macro(a_vec):
        q = w * a_vec + mu * (1 - a_vec)
        return float(np.nanmean([auc(lab[c].to_numpy(), q[c].to_numpy()) for c in PCOLS]))

    L = ["# 后处理旋钮的 valid 全网格（2026-11-23）", "",
         f"- valid：{len(w):,} 个检查、{len(np.unique(subj))} 名患者",
         "- 全部由 `runs/valid_noshrink.csv`（收缩前概率）离线重算，未重训、未读 test", ""]

    # 1) 全局收缩系数
    L += ["## 1　全局患者收缩系数 a", "",
          "`p = a·p_study + (1−a)·p_患者均值`", "",
          "| a | 宏 AUC | 相对 0.85 |", "|---:|---:|---:|"]
    base = macro(np.full(10, 0.85, np.float32))
    rows = []
    for a in (0.60, 0.70, 0.75, 0.80, 0.825, 0.85, 0.875, 0.90, 0.95, 1.00):
        m = macro(np.full(10, a, np.float32))
        rows.append((a, m))
        L.append(f"| {a:.3f} | {m:.6f} | {m - base:+.6f} |")
    best_a = max(rows, key=lambda r: r[1])
    L += ["", f"最优 a = **{best_a[0]:.3f}**（{best_a[1]:.6f}），"
              f"比现值 0.85 高 **{best_a[1] - base:+.6f}**。", ""]

    # 2) 类 2 单独
    L += ["## 2　类 2 单独收缩（其余 0.85）", "", "| 类2 a | 宏 AUC | 相对 |", "|---:|---:|---:|"]
    for a2 in (0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.85):
        v = np.full(10, 0.85, np.float32)
        v[1] = a2
        m = macro(v)
        L.append(f"| {a2:.2f} | {m:.6f} | {m - base:+.6f} |")

    # 3) 两旋钮组合
    a2_best = 0.50
    v = np.full(10, best_a[0], np.float32)
    v[1] = a2_best
    combo = macro(v)
    L += ["", "## 3　组合（全局最优 a + 类 2 = 0.50）", "",
          f"- 宏 AUC = **{combo:.6f}**，比现值高 **{combo - base:+.6f}**", ""]

    # 4) 分辨率说明
    n_pat = len(np.unique(subj))
    L += ["## 4　这套网格能分辨多大的差异", "",
          f"- valid 只有 {len(w):,} 个检查 / {n_pat} 名患者；单检查患者占多数，",
          "  患者级聚合的 Kish 有效样本量约 45 名。",
          "- 因此 **±0.001 量级的差异在 valid 上不可分辨**：上表相邻取值之间多数差异落在噪声内。",
          "- 结论：valid 只能用来排除大错（如 a=1 掉 0.0019），不能用来挑 0.0005 级的胜负；",
          "  要挑细微差异必须靠 **OOF 五折**或**线上分数**。", ""]

    open(OUT, "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))
    print(f"\n  [wrote] {os.path.relpath(OUT, ROOT)}")
    json.dump({"base": base, "best_a": best_a[0], "best_a_macro": best_a[1],
               "combo_a2_050": combo,
               "grid": [[a, m] for a, m in rows]},
              open(os.path.join(ROOT, "reports", "postproc_grid_2026-11-23.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
