"""按预测能力给 10 类排序，以五折 OOF 为主口径、官方 valid 为对照。

为什么以 OOF 为主：valid 只有 859 个 study，类 1 只有 24 个正样本，它的逐类估计不可靠
（患者级 CI 宽达 0.25）。OOF 覆盖 18,513 个 study，类 1 的正样本约 570 个。

输出
----
  reports/弱类排序_2026-11-19.md   排名表 + 置信区间 + 逐折稳定性
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = r"E:\挑战赛"
PCOLS = list(range(1, 11))
NAMES = {1: "Atelectasis", 2: "Cardiomegaly", 3: "Effusion", 4: "Infiltration",
         5: "Mass/Nodule", 6: "Pneumonia", 7: "Pneumothorax", 8: "Consolidation",
         9: "Edema", 10: "No Finding"}

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


def study_labels(csv_path: str) -> pd.DataFrame:
    d = pd.read_csv(csv_path)
    Y = np.stack([parse_label(s) for s in d.Predict_class])
    lab = pd.DataFrame(Y, columns=PCOLS)
    lab["Subject_id"] = d.Subject_id.values
    lab["Study_id"] = d.Study_id.values
    return lab.groupby(["Subject_id", "Study_id"])[PCOLS].max()


def wide(path: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    return d.pivot_table(index=["Subject_id", "Study_id"], columns="Predict_class",
                         values="Probability", aggfunc="first").reindex(columns=PCOLS)


def main() -> None:
    # ---- OOF ----
    oof_lab = study_labels(os.path.join(ROOT, "cxr_first", "train.csv"))
    parts = []
    for f in range(5):
        p = os.path.join(ROOT, "runs", f"oof_f{f}_study.csv")
        if not os.path.exists(p):
            raise SystemExit(f"缺少 {p}")
        parts.append(wide(p))
    oof = pd.concat(parts)
    oof = oof[~oof.index.duplicated()]
    idx = oof.index.intersection(oof_lab.index)
    oof, oof_lab = oof.loc[idx], oof_lab.loc[idx]
    print(f"  OOF: {len(oof)} 个 study，{oof.index.get_level_values(0).nunique()} 名患者")

    # ---- 官方 valid ----
    va_lab = study_labels(os.path.join(ROOT, "cxr_first", "valid.csv"))
    va = wide(os.path.join(ROOT, "runs", "orig_valid.csv"))
    vidx = va.index.intersection(va_lab.index)
    va, va_lab = va.loc[vidx], va_lab.loc[vidx]

    # ---- 逐折稳定性 ----
    fold_tab = pd.read_csv(os.path.join(ROOT, "04_新一轮", "00_规则", "patient_folds_5f.csv"))
    per_fold = {c: [] for c in PCOLS}
    for f in range(5):
        ids = set(fold_tab[fold_tab.fold == f].Study_id.astype(str))
        sub = oof[[str(s) in ids for s in oof.index.get_level_values(1)]]
        for c in PCOLS:
            per_fold[c].append(auc(oof_lab.loc[sub.index, c].to_numpy(), sub[c].to_numpy()))

    # ---- 患者级 bootstrap CI（200 次） ----
    rng = np.random.default_rng(0)
    subj = oof.index.get_level_values(0).to_numpy()
    uniq = np.unique(subj)
    boot = {c: [] for c in PCOLS}
    # 预建 患者 -> 行号 索引，避免每次重采都全表扫描（否则是 O(患者²)）
    rows_of: dict[str, np.ndarray] = {}
    for u, grp in pd.Series(np.arange(len(subj)), index=subj).groupby(level=0):
        rows_of[u] = grp.to_numpy()
    lab_np = {c: oof_lab[c].to_numpy() for c in PCOLS}
    prd_np = {c: oof[c].to_numpy() for c in PCOLS}
    for _ in range(200):
        pick = rng.choice(uniq, len(uniq), replace=True)
        take = np.concatenate([rows_of[u] for u in pick])
        for c in PCOLS:
            boot[c].append(auc(lab_np[c][take], prd_np[c][take]))
    ci = {c: np.percentile(boot[c], [2.5, 97.5]) for c in PCOLS}

    rows = []
    for c in PCOLS:
        a = auc(oof_lab[c].to_numpy(), oof[c].to_numpy())
        n_st = int((oof_lab[c] > 0).sum())
        n_pt = int(oof_lab.loc[oof_lab[c] > 0].index.get_level_values(0).nunique())
        av = auc(va_lab[c].to_numpy(), va[c].to_numpy())
        rows.append({
            "class": c, "name": NAMES[c], "oof_auc": a,
            "ci_lo": ci[c][0], "ci_hi": ci[c][1], "ci_width": ci[c][1] - ci[c][0],
            "n_study": n_st, "n_patient": n_pt,
            "fold_min": min(per_fold[c]), "fold_max": max(per_fold[c]),
            "valid_auc": av, "n_valid": int((va_lab[c] > 0).sum()),
        })
    rows.sort(key=lambda r: r["oof_auc"])

    lines = ["# 弱类排序（2026-11-19）", "",
             f"主口径：五折 OOF，{len(oof):,} 个 study / "
             f"{oof.index.get_level_values(0).nunique():,} 名患者（单模型、no-KD、固定 last12）。",
             f"对照：官方 valid，{len(va)} 个 study（交付版四成员融合）。",
             "排序依据为 OOF 的检查级 AUC，由弱到强。", "",
             "| # | 类别 | **OOF AUC** | 95% 区间 | 区间宽 | OOF 正样本（study/患者） | 折间范围 | valid AUC | valid 正样本 |",
             "|---:|---|---:|---|---:|---|---|---:|---:|"]
    for i, r in enumerate(rows, 1):
        lines.append("| {} | {} {} | **{:.3f}** | [{:.3f}, {:.3f}] | {:.3f} | {} / {} | "
                     "{:.3f}–{:.3f} | {:.3f} | {} |".format(
                         i, r["class"], r["name"], r["oof_auc"], r["ci_lo"], r["ci_hi"],
                         r["ci_width"], r["n_study"], r["n_patient"],
                         r["fold_min"], r["fold_max"], r["valid_auc"], r["n_valid"]))
    lines.append("")
    lines += ["## 怎么看", "",
              "- **折间范围**是 5 折各自留出折上的 AUC 最小/最大值。范围窄说明这一类学得稳；范围宽说明结论依赖数据划分。",
              "- **valid 列**只作对照。它的正样本极少的类别（类 1 只有 24 个）估计不可靠，不可用来排名。",
              "- 区间来自患者级 bootstrap 200 次（整患者重采样，避免同一患者多张图被当独立样本）。", ""]
    out = os.path.join(ROOT, "reports", "弱类排序_2026-11-19.md")
    open(out, "w", encoding="utf-8").write("\n".join(lines))
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "reports", "弱类排序_2026-11-19.csv"),
                              index=False, encoding="utf-8-sig")

    print()
    print("  %-2s %-18s %8s %-17s %6s %-13s %-15s %8s" % (
        "#", "类别", "OOF", "95% 区间", "范围", "折间范围", "valid(正样本)", ""))
    for i, r in enumerate(rows, 1):
        print("  %-2d %-18s %8.3f [%.3f,%.3f] %6.3f %.3f–%.3f   valid %.3f (%d)" % (
            i, f'{r["class"]} {r["name"]}', r["oof_auc"], r["ci_lo"], r["ci_hi"],
            r["ci_width"], r["fold_min"], r["fold_max"], r["valid_auc"], r["n_valid"]))
    print()
    print(f"  [wrote] {out}")


if __name__ == "__main__":
    main()
