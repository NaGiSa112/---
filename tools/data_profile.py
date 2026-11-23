"""Data profile: per-class patient counts and images-per-patient (multi-view) structure.

Answers the two questions a competition plan needs before anything else:

  1. 每个类别有多少**患者**是阳性（不是图片数、不是 study 数）
  2. 每个患者大约有多少张图，以及这些图是怎么分布的（多 study / 多体位）

Writes reports/数据画像_2026-11-18.md and .csv
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
DATA = os.path.join(ROOT, "cxr_first")
NAMES = {1: "Atelectasis", 2: "Cardiomegaly", 3: "Effusion", 4: "Infiltration",
         5: "Mass/Nodule", 6: "Pneumonia", 7: "Pneumothorax", 8: "Consolidation",
         9: "Edema", 10: "No Finding"}


def parse_label(s) -> np.ndarray:
    v = np.zeros(10, np.int8)
    s = str(s).strip()
    if s == "" or s.lower() == "nan":
        return v
    for c in s.split(";"):
        c = c.strip()
        if c:
            v[int(c) - 1] = 1
    return v


def load(split: str) -> pd.DataFrame:
    d = pd.read_csv(os.path.join(DATA, f"{split}.csv"))
    Y = np.stack([parse_label(s) for s in d.Predict_class]) if "Predict_class" in d else None
    if Y is not None:
        for k in range(10):
            d[f"c{k+1}"] = Y[:, k]
    return d


def main() -> None:
    rows, lines = [], []
    splits = {s: load(s) for s in ("train", "valid", "test")}

    lines += ["# 数据画像：每类的患者数与每患者图片数", "",
              "统计口径：**患者 = Subject_id 去重**；一个患者只要任一检查带该标签即为阳性。", ""]

    print("  ① 三个 split 的规模")
    print("  %-8s %8s %8s %8s %10s %10s" % ("split", "图片", "study", "患者", "每患者图片", "每 study 图片"))
    lines += ["## ① 规模", "",
              "| split | 图片 | study | 患者 | 每患者图片（均值） | 每治疗 study 图片 |",
              "|---|---:|---:|---:|---:|---:|"]
    for s, d in splits.items():
        n_img, n_st, n_sub = len(d), d.Study_id.nunique(), d.Subject_id.nunique()
        per_sub = d.groupby("Subject_id").size().mean()
        per_st = d.groupby("Study_id").size().mean()
        print("  %-8s %8d %8d %8d %10.2f %10.2f" % (s, n_img, n_st, n_sub, per_sub, per_st))
        lines.append(f"| {s} | {n_img:,} | {n_st:,} | {n_sub:,} | {per_sub:.2f} | {per_st:.2f} |")

    print()
    print("  ② 每类的**患者**数量（这是最有用的口径）")
    lines += ["", "## ② 每类阳性患者数", "",
              "| 类别 | train 患者 | valid 患者 | test 患者 | train 阳性率 | valid 阳性率 |",
              "|---:|---:|---:|---:|---:|---:|"]
    print("  %-18s %10s %10s %10s %10s" % ("类别", "train", "valid", "test", "train率"))
    for k in range(1, 11):
        vals = []
        for s in ("train", "valid", "test"):
            d = splits[s]
            if f"c{k}" in d:
                vals.append(int(d.groupby("Subject_id")[f"c{k}"].max().sum()))
            else:
                vals.append(-1)  # test 无标签
        n_tr = splits["train"].Subject_id.nunique()
        n_va = splits["valid"].Subject_id.nunique()
        r_tr = vals[0] / n_tr * 100
        r_va = vals[1] / n_va * 100
        print("  %-18s %10d %10d %10s %9.1f%%" % (
            f"{k} {NAMES[k]}", vals[0], vals[1],
            vals[2] if vals[2] >= 0 else "无标签", r_tr))
        lines.append(f"| {k} {NAMES[k]} | {vals[0]:,} | {vals[1]:,} | "
                     f"{'—' if vals[2] < 0 else vals[2]} | {r_tr:.1f}% | {r_va:.1f}% |")
        rows.append({"class": k, "name": NAMES[k],
                     "train_patients": vals[0], "valid_patients": vals[1],
                     "train_rate": round(r_tr, 2), "valid_rate": round(r_va, 2)})

    print()
    print("  ③ 每患者图片数的分布（多角度 = 多 study × 每 study 多体位）")
    lines += ["", "## ③ 每患者的图片数分布", "",
              "| split | 均值 | 中位 | P90 | 最大 | 只有1张 | 只有1个study |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    print("  %-8s %6s %6s %6s %6s %8s %10s" % ("split", "均值", "中位", "P90", "最大", "1张", "1个study"))
    for s, d in splits.items():
        per = d.groupby("Subject_id").size()
        per_st = d.groupby("Subject_id").Study_id.nunique()
        print("  %-8s %6.2f %6d %6d %6d %7.1f%% %9.1f%%" % (
            s, per.mean(), per.median(), per.quantile(0.9), per.max(),
            100 * (per == 1).mean(), 100 * (per_st == 1).mean()))
        lines.append(f"| {s} | {per.mean():.2f} | {int(per.median())} | "
                     f"{int(per.quantile(0.9))} | {int(per.max())} | "
                     f"{100*(per==1).mean():.1f}% | {100*(per_st==1).mean():.1f}% |")

    print()
    print("  ④ 曝光体位构成（多角度到底是什么角度）")
    lines += ["", "## ④ 体位构成", "",
              "| split | PA | AP | LATERAL | LL | 其它 | 每 study 平均体位数 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    print("  %-8s %7s %7s %9s %7s %7s %10s" % ("split", "PA", "AP", "LATERAL", "LL", "其它", "每study体位"))
    for s, d in splits.items():
        vc = d.ViewPosition.fillna("NA").value_counts()
        tot = vc.sum()
        others = tot - sum(vc.get(k, 0) for k in ("PA", "AP", "LATERAL", "LL"))
        per_st_views = d.groupby("Study_id").ViewPosition.nunique().mean()
        print("  %-8s %6.1f%% %6.1f%% %8.1f%% %6.1f%% %6.1f%% %10.2f" % (
            s, 100 * vc.get("PA", 0) / tot, 100 * vc.get("AP", 0) / tot,
            100 * vc.get("LATERAL", 0) / tot, 100 * vc.get("LL", 0) / tot,
            100 * others / tot, per_st_views))
        lines.append(f"| {s} | {100*vc.get('PA',0)/tot:.1f}% | {100*vc.get('AP',0)/tot:.1f}% | "
                     f"{100*vc.get('LATERAL',0)/tot:.1f}% | {100*vc.get('LL',0)/tot:.1f}% | "
                     f"{100*others/tot:.1f}% | {per_st_views:.2f} |")

    print()
    print("  ⑤ 弱类的阳性患者也能凑够样本吗（决定能不能做交叉验证）")
    lines += ["", "## ⑤ 弱类的阳性患者数够不够用", "",
              "| 类别 | 全部阳性患者（train+valid） | 够不够做 5 折 |",
              "|---:|---:|---|"]
    print("  %-18s %12s   判定" % ("类别", "train+valid"))
    for k in (1, 4, 6, 8):
        n = 0
        for s in ("train", "valid"):
            d = splits[s]
            n += int(d.groupby("Subject_id")[f"c{k}"].max().sum())
        ok = "够（每折约 %d）" % (n // 5) if n >= 250 else "偏少（每折约 %d）" % (n // 5)
        print("  %-18s %12d   %s" % (f"{k} {NAMES[k]}", n, ok))
        lines.append(f"| {k} {NAMES[k]} | {n:,} | {ok} |")

    out_md = os.path.join(ROOT, "reports", "数据画像_2026-11-18.md")
    open(out_md, "w", encoding="utf-8").write("\n".join(lines))
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "reports", "数据画像_2026-11-18.csv"),
                              index=False, encoding="utf-8-sig")
    print()
    print(f"  [wrote] {out_md}")


if __name__ == "__main__":
    main()
