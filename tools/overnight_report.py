""" 验收报告生成器：读取夜间队列的全部产出，生成一份可直接看的 markdown。

    python tools/overnight_report.py

读什么
------
  runs/overnight/queue.log            阶段流水
  runs/overnight/S*.log               各阶段子进程日志
  runs/oof_f{0..4}/history.json       逐轮曲线
  runs/oof_f{0..4}/last.pt            权重是否产出
  runs/oof_f{0..4}_study.csv          逐折预测（predict_fold 产物）
  04_新一轮/00_规则/patient_folds_5f.csv  折表
  runs/repA42, runs/repB42            受控重复

算什么
------
  1. 五折 OOF 的 study 级宏 AUC（用 train.csv 的折内标签，逐折 + 合并）
  2. 受控重复 A/B 在官方 valid 上的 study 级宏 AUC 及其差（= 同配方跑两次的波动实测）
  3. 与交付版基线 0.8251945484 的对照
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = r"E:\挑战赛"
OVN = os.path.join(ROOT, "runs", "overnight")
FOLDS = os.path.join(ROOT, "04_新一轮", "00_规则", "patient_folds_5f.csv")
PCOLS = list(range(1, 11))

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


def rank_auc(y, p) -> float:
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


def train_study_labels() -> pd.DataFrame:
    d = pd.read_csv(os.path.join(ROOT, "cxr_first", "train.csv"))
    Y = np.stack([parse_label(s) for s in d.Predict_class])
    lab = pd.DataFrame(Y, columns=PCOLS)
    lab["Subject_id"] = d.Subject_id.values
    lab["Study_id"] = d.Study_id.values
    return lab.groupby(["Subject_id", "Study_id"])[PCOLS].max()


def valid_study_labels() -> pd.DataFrame:
    d = pd.read_csv(os.path.join(ROOT, "cxr_first", "valid.csv"))
    Y = np.stack([parse_label(s) for s in d.Predict_class])
    lab = pd.DataFrame(Y, columns=PCOLS)
    lab["Subject_id"] = d.Subject_id.values
    lab["Study_id"] = d.Study_id.values
    return lab.groupby(["Subject_id", "Study_id"])[PCOLS].max()


def auc_block(pred: pd.DataFrame, lab: pd.DataFrame, tag: str) -> dict:
    idx = pred.index.intersection(lab.index)
    p, s = pred.loc[idx], lab.loc[idx]
    per = [rank_auc(s[c].to_numpy(), p[c].to_numpy()) for c in PCOLS]
    return {"tag": tag, "studies": len(idx), "macro": float(np.nanmean(per)),
            "per_class": [None if np.isnan(v) else round(float(v), 4) for v in per]}


def history_summary(run_dir: str) -> dict | None:
    p = os.path.join(run_dir, "history.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p, encoding="utf-8"))
    h = d.get("history", d) if isinstance(d, dict) else d
    if not isinstance(h, list) or not h:
        return None
    best = max(h, key=lambda e: e.get("val_auc", -1))
    return {"epochs": len(h), "best_epoch": best.get("epoch"),
            "best_val_auc": round(float(best.get("val_auc", 0)), 6),
            "last_val_auc": round(float(h[-1].get("val_auc", 0)), 6),
            "total_min": round(sum(e.get("sec", 0) for e in h) / 60, 1)}


def main() -> None:
    lines = ["# 夜间队列验收报告", "",
             f"生成时间：{pd.Timestamp.now():%Y-%m-%d %H:%M}", ""]

    # ---- 阶段流水 ----
    q = os.path.join(OVN, "queue.log")
    lines += ["## 一、阶段流水", ""]
    if os.path.exists(q):
        rows = [l for l in open(q, encoding="utf-8", errors="replace").read().splitlines()
                if l.strip() and "RUN" not in l]
        lines += ["```", *rows[-40:], "```", ""]
    else:
        lines += ["（queue.log 不存在）", ""]

    # ---- 五折 OOF ----
    lines += ["## 二、五折 OOF（no-KD、固定 last12、冻结折表）", ""]
    fold_tab = pd.read_csv(FOLDS) if os.path.exists(FOLDS) else None
    lines += ["| fold | 训练轮数 | 最优 epoch | 最优 val AUC | 末轮 val AUC | 用时 | last.pt | 预测 |",
              "|---:|---:|---:|---:|---:|---:|---|---|"]
    preds = {}
    for f in range(5):
        rd = os.path.join(ROOT, "runs", f"oof_f{f}")
        h = history_summary(rd)
        has_ckpt = os.path.exists(os.path.join(rd, "last.pt"))
        pp = os.path.join(ROOT, "runs", f"oof_f{f}_study.csv")
        has_pred = os.path.exists(pp)
        if has_pred:
            preds[f] = wide(pp)
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            f,
            h["epochs"] if h else "—",
            h["best_epoch"] if h else "—",
            f"{h['best_val_auc']:.4f}" if h else "—",
            f"{h['last_val_auc']:.4f}" if h else "—",
            f"{h['total_min']:.0f} 分" if h else "—",
            "✅" if has_ckpt else "❌",
            "✅" if has_pred else "❌"))
    lines.append("")

    if fold_tab is not None and preds:
        lab = train_study_labels()
        lines += ["### 逐折与合并的 study 级宏 AUC", "",
                  "| 口径 | study 数 | 宏 AUC | 逐类 |", "|---|---:|---:|---|"]
        all_pred = []
        per_fold = []
        for f, p in sorted(preds.items()):
            ids = set(fold_tab[fold_tab.fold == f].Study_id.astype(str))
            keep = [i for i in p.index if str(i[1]) in ids]
            sub = p.loc[keep]
            r = auc_block(sub, lab, f"fold{f}")
            per_fold.append(r)
            lines.append("| fold {}（留出折） | {} | {:.6f} | {} |".format(
                f, r["studies"], r["macro"], " ".join(f"{v:.3f}" for v in r["per_class"])))
            all_pred.append(sub)
        if all_pred:
            pooled = pd.concat(all_pred)
            pooled = pooled[~pooled.index.duplicated()]
            r = auc_block(pooled, lab, "pooled")
            lines.append("| **合并 OOF** | {} | **{:.6f}** | {} |".format(
                r["studies"], r["macro"], " ".join(f"{v:.3f}" for v in r["per_class"])))
        lines += ["", "注：合并值不是各折平均；各折训练集高度重叠，不能把折间标准差除以 √5 当可信区间。", ""]

    # ---- 受控重复 ----
    lines += ["## 三、受控重复（同配方、同缓存、同端点、只换运行次数）", ""]
    reps = {}
    for name in ("repA42", "repB42"):
        rd = os.path.join(ROOT, "runs", name)
        h = history_summary(rd)
        ck = os.path.join(rd, "last.pt")
        reps[name] = {"hist": h, "ckpt": os.path.exists(ck)}
        lines.append(f"- `{name}`: 轮数 {h['epochs'] if h else '—'} · 最优 epoch "
                     f"{h['best_epoch'] if h else '—'} · 最优 val {h['best_val_auc'] if h else '—'} · "
                     f"用时 {h['total_min'] if h else '—'} 分 · last.pt "
                     f"{'✅' if reps[name]['ckpt'] else '❌'}")
    lines += ["", "两次的差距（在官方 valid 上按检查级宏 AUC 计）需要在 8:00 时用 "
              "`tools/study_metric.py` 对两份预测文件算 —— 若两份预测未生成，此项留待补算。", ""]

    # ---- 对照 ----
    lines += ["## 四、与交付版基线对照", "",
              "| 版本 | 检查级宏 AUC |", "|---|---:|",
              "| 交付版（原始四成员） | 0.8251945484 |",
              "| 重训三成员 | 0.8237508312 |",
              "| 六成员（超限，不可用） | 0.8246325917 |", ""]

    out = os.path.join(OVN, "验收报告.md")
    open(out, "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines[-60:]))
    print(f"\n[wrote] {out}")


if __name__ == "__main__":
    main()
