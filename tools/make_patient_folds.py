"""Freeze a patient-grouped 5-fold table for cross-validation.

Why this is a separate artefact
-------------------------------
`src/cxr/train.py::patient_fold_split(..., seed=cfg.seed)` derives the folds from the
*training* seed. Changing the seed therefore changes which patients are held out, so two
runs with different seeds are not a paired comparison on the same data.

This script writes ONE fold table, keyed by Study_id, that every future run reuses via
`--fold-csv`. The split seed here is fixed and unrelated to any training seed.

Balancing
---------
Greedy assignment on patient-level features:
  - the 10-dim patient label vector (a patient is positive for a class if any study is)
  - studies-per-patient bucket (1 / 2-4 / 5-12 / 13+)
Patients are sorted by "difficulty" (number of positive classes, then study count) and sent
to whichever fold current carries the least of that feature mass. One patient never splits.

Outputs
-------
  04_新一轮/00_规则/patient_folds_5f.csv     Study_id, Subject_id, fold
  04_新一轮/00_规则/patient_folds_5f.json    per-fold composition + integrity checks
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = r"E:\挑战赛"
OUT_DIR = os.path.join(ROOT, "04_新一轮", "00_规则")
N_FOLDS = 5
SEED = 2026

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


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    tr = pd.read_csv(os.path.join(ROOT, "cxr_first", "train.csv"))
    Y = np.stack([parse_label(s) for s in tr.Predict_class])
    lab = pd.DataFrame(Y, columns=[f"c{k}" for k in range(1, 11)])
    lab["Subject_id"] = tr.Subject_id.values
    lab["Study_id"] = tr.Study_id.values

    # 患者级标签与检查次数
    pat = lab.groupby("Subject_id")[lab.columns[:10]].max()
    n_study = lab.groupby("Subject_id").Study_id.nunique()
    pat["n_study"] = n_study

    def bucket(n: int) -> int:
        return 0 if n == 1 else 1 if n <= 4 else 2 if n <= 12 else 3

    pat["bucket"] = [bucket(int(n)) for n in pat.n_study]
    pat["difficulty"] = pat[lab.columns[:10]].sum(axis=1)

    rng = np.random.default_rng(SEED)
    patients = pat.index.to_numpy()
    # 难的先分，同级内随机（种子固定）
    order = sorted(patients, key=lambda p: (-pat.loc[p, "difficulty"], -pat.loc[p, "n_study"],
                                           rng.random()))

    # 每个 fold 累积：每类的阳性患者数 + 患者数 + 图像数 + bucket 计数
    fold_cls = np.zeros((N_FOLDS, 10))
    fold_n_pat = np.zeros(N_FOLDS)
    fold_n_img = np.zeros(N_FOLDS)
    fold_bucket = np.zeros((N_FOLDS, 4))
    assign = {}
    for p in order:
        vec = pat.loc[p, lab.columns[:10]].to_numpy(float)
        b = int(pat.loc[p, "bucket"])
        n_img = float(pat.loc[p, "n_study"])
        # 目标：让每个 fold 的类阳性质量尽量均衡；难度高的优先填最空的
        tot = fold_n_pat.sum()
        score = (fold_cls @ vec) / (fold_cls.sum(axis=1) + 1.0) + (fold_n_pat / tot if tot > 0 else 0.0)
        f = int(np.argmin(score))
        assign[p] = f
        fold_cls[f] += vec
        fold_n_pat[f] += 1
        fold_n_img[f] += n_img
        fold_bucket[f, b] += 1

    lab["fold"] = lab.Subject_id.map(assign).astype(int)
    # 一个 study 多张图，折表按 study 去重（train.py 用 Study_id -> fold 查表）
    out = (lab.groupby(["Study_id", "Subject_id"], as_index=False).fold.first())
    csv_path = os.path.join(OUT_DIR, "patient_folds_5f.csv")
    out.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # ---- 完整性检查 ----
    checks = {}
    checks["studies_total"] = int(len(out))
    checks["studies_unique"] = int(out.Study_id.nunique())
    checks["every_study_once"] = bool(len(out) == out.Study_id.nunique())
    per_pat_folds = out.groupby("Subject_id").fold.nunique()
    checks["patients_split_across_folds"] = int((per_pat_folds > 1).sum())
    checks["patients_total"] = int(out.Subject_id.nunique())
    folds = sorted(out.fold.unique().tolist())
    checks["fold_ids"] = folds
    checks["all_folds_present"] = folds == list(range(N_FOLDS))

    report = {"seed": SEED, "n_folds": N_FOLDS, "checks": checks, "per_fold": []}
    print(f"  训练集：{checks['studies_total']} study / {checks['patients_total']} 患者")
    print(f"  %-5s %7s %8s %9s %s" % ("fold", "study", "患者", "图像", "每类阳性患者"))
    for f in range(N_FOLDS):
        sub = out[out.fold == f]
        st_ids = set(sub.Study_id)
        imgs = int(tr[tr.Study_id.isin(st_ids)].shape[0])   # 图像数（含 multi-image study）
        pos = [int(lab[lab.Study_id.isin(st_ids)].groupby("Subject_id")[c].max().sum())
               for c in lab.columns[:10]]
        print("  %-5d %7d %8d %9d %s" % (f, len(sub), sub.Subject_id.nunique(), imgs,
                                         " ".join(f"{v:4d}" for v in pos)))
        report["per_fold"].append({"fold": f, "studies": len(sub),
                                  "patients": int(sub.Subject_id.nunique()),
                                  "images": imgs, "pos_patients": pos})
    print()
    print("  完整性：")
    for k in ("every_study_once", "all_folds_present"):
        print(f"    {k}: {'✅' if checks[k] else '❌'}")
    print(f"    患者跨折数（必须为 0）: {checks['patients_split_across_folds']} "
          f"{'✅' if checks['patients_split_across_folds'] == 0 else '❌'}")

    json_path = os.path.join(OUT_DIR, "patient_folds_5f.json")
    json.dump(report, open(json_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n  [wrote] {csv_path}")
    print(f"  [wrote] {json_path}")

    if not checks["every_study_once"] or checks["patients_split_across_folds"] or \
            not checks["all_folds_present"]:
        raise SystemExit("折表完整性检查未通过")


if __name__ == "__main__":
    main()
