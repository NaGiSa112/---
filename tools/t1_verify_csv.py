"""T2 §2.1 验收：逐图概率 CSV 是否严格合规。

只检查**格式与键集**，不碰模型、不做任何聚合 —— 聚合由队长统一算。
通过条件：
- 行数等于目标集（valid 1422 / fold2 6150）
- Dicom_id 与官方 csv 的对应集合**不重不漏**（逐键比较，不是取交集）
- 列名恰为 Dicom_id,Subject_id,Study_id,p1..p10
- p1..p10 全部有限且严格落在 (0,1)

    python tools/t1_verify_csv.py --probs runs_t1/hrnet18_384_f2_accum2/fold2_probs.csv --split fold2
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PCOLS = [f"p{k}" for k in range(1, 11)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True)
    ap.add_argument("--split", default="fold2", choices=["fold2", "valid"])
    ap.add_argument("--fold-csv", default=os.path.join(ROOT, "data", "patient_folds_5f.csv"))
    a = ap.parse_args()

    data_dir = os.environ.get("CXR_DATA", "")
    if a.split == "valid":
        exp = pd.read_csv(os.path.join(data_dir, "valid.csv"))
    else:
        tr = pd.read_csv(os.path.join(data_dir, "train.csv"))
        fk = pd.read_csv(a.fold_csv)
        fmap = dict(zip(fk.Study_id.astype(str), fk.fold.astype(int)))
        exp = tr[tr.Study_id.astype(str).map(fmap) == 2]
    exp_ids = exp["Dicom_id"].astype(str).tolist()

    df = pd.read_csv(a.probs)
    errs = []
    if list(df.columns) != ["Dicom_id", "Subject_id", "Study_id"] + PCOLS:
        errs.append(f"列名不符: {list(df.columns)}")
    if len(df) != len(exp_ids):
        errs.append(f"行数 {len(df)} != 期望 {len(exp_ids)}")
    ids = df["Dicom_id"].astype(str).tolist()
    if len(set(ids)) != len(ids):
        errs.append(f"Dicom_id 有重复 {len(ids) - len(set(ids))} 个")
    miss = set(exp_ids) - set(ids)
    extra = set(ids) - set(exp_ids)
    if miss or extra:
        errs.append(f"键集不符: 缺失 {len(miss)} 个 / 多余 {len(extra)} 个 "
                    f"(例: {list(miss)[:3]} / {list(extra)[:3]})")
    # docs/56 §3 要求核对的「图像 ID 与患者、检查的完整对应」：
    # 逐行比较 (Subject_id, Study_id) 是否与官方元数据一致，而不只是 Dicom_id 集合相等。
    if not (miss or extra) and len(df) == len(exp_ids):
        ref = exp.assign(_k=exp.Dicom_id.astype(str)).set_index("_k")[["Subject_id", "Study_id"]]
        got = df.assign(_k=df.Dicom_id.astype(str)).set_index("_k")[["Subject_id", "Study_id"]]
        j = ref.join(got, how="left", lsuffix="_ref", rsuffix="_csv")
        bad = ((j.Subject_id_ref.astype(str) != j.Subject_id_csv.astype(str)) |
               (j.Study_id_ref.astype(str) != j.Study_id_csv.astype(str)))
        if bad.any():
            ex = j[bad].head(3)
            errs.append(f"{int(bad.sum())} 行的患者/检查与官方元数据不一致，例: "
                        f"{ex.to_dict(orient='index')}")
    if any(c in df.columns for c in PCOLS):
        P = df[PCOLS].to_numpy(np.float64)
        if not np.isfinite(P).all():
            errs.append("存在非有限值")
        if P.min() <= 0 or P.max() >= 1:
            errs.append(f"概率越界: min={P.min():.3e} max={P.max():.3e}")
    sha = hashlib.sha256(open(a.probs, "rb").read()).hexdigest()
    size = os.path.getsize(a.probs)

    print(f"[verify] {a.probs}")
    print(f"[verify] rows={len(df)}  expected={len(exp_ids)}  "
          f"studies={df.Study_id.nunique()}  sha256={sha}  bytes={size}")
    if errs:
        for e in errs:
            print(f"[FAIL] {e}")
        raise SystemExit(1)
    print("[verify] 通过：键集不重不漏，p1..p10 ∈ (0,1) 且有限")


if __name__ == "__main__":
    main()
