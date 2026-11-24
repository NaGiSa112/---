"""用交付包内的 test_probs.csv 精确重建 submission.csv，并生成后处理变体。

为什么这样做
------------
包内 `reports/test_probs.csv` 是**逐图融合后**的概率（2,218 行 × p1..p10），
`model/submission.csv` 是最终提交文件。若能用前者重建后者（逐位一致），
就说明后处理链条完全可复现，之后任何变体都可以秒级生成且与官方口径一致。

变体维度（都不需要重跑模型）
    --shrink a            全局患者收缩
    --shrink-by-class     逐类覆盖，如 "2:0.45"
    --lat-weight w        侧位权重
    --tag                 输出文件名

    python tools/bank_variants.py --check
    python tools/bank_variants.py --shrink 0.85 --shrink-by-class 2:0.45 --tag V1_class2
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd

ROOT = r"E:\挑战赛"
PKG = os.path.join(ROOT, "_pkgcheck", "陈思阳_最终提交材料")
BANK = os.path.join(PKG, "reports", "test_probs.csv")
SHIPPED = os.path.join(PKG, "model", "submission.csv")
OUTDIR = os.path.join(ROOT, "submissions", "ladder_2026-11-23")
PCOLS = [f"p{k}" for k in range(1, 11)]

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def build(bank: pd.DataFrame, lat_weight: float, shrink: float,
          shrink_by_class: dict[str, float] | None = None,
          round_to: int = 6) -> pd.DataFrame:
    """与 src/cxr/infer.py::write_submission 完全相同的后处理。"""
    d = bank.copy()
    if "ViewPosition" in d.columns and lat_weight < 1.0:
        w = np.where(np.isin(d["ViewPosition"].fillna("NA").values, ["LATERAL", "LL"]),
                     lat_weight, 1.0)
    else:
        w = np.ones(len(d))
    for c in PCOLS:
        d[c] = d[c] * w
    g = d.groupby(["Subject_id", "Study_id"], as_index=False)[PCOLS].sum()
    wsum = pd.Series(w).groupby([d["Subject_id"].values, d["Study_id"].values]).sum()
    g[PCOLS] = g[PCOLS].div(wsum.values[:, None])

    a_vec = np.full(10, float(shrink))
    for k, a in (shrink_by_class or {}).items():
        a_vec[int(k) - 1] = float(a)
    if (a_vec < 1.0).any():
        g0 = g[PCOLS].values
        pm0 = g.groupby("Subject_id")[PCOLS].transform("mean").values
        g[PCOLS] = a_vec[None, :] * g0 + (1.0 - a_vec[None, :]) * pm0

    long = g.melt(id_vars=["Subject_id", "Study_id"], value_vars=PCOLS,
                  var_name="cls", value_name="Probability")
    long["Predict_class"] = long["cls"].str.lstrip("p").astype(int)
    long["Probability"] = long["Probability"].astype(float).round(round_to)
    return long[["Subject_id", "Study_id", "Predict_class", "Probability"]] \
        .sort_values(["Subject_id", "Study_id", "Predict_class"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--lat-weight", type=float, default=0.6)
    ap.add_argument("--shrink", type=float, default=0.85)
    ap.add_argument("--shrink-by-class", default="")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    bank = pd.read_csv(BANK)
    # 概率库没有 ViewPosition，从官方 test.csv 按 Dicom_id 补上（侧位权重需要它）
    if "ViewPosition" not in bank.columns:
        for cand in (os.path.join(PKG, "cxr_first", "test.csv"),
                     os.path.join(ROOT, "cxr_first", "test.csv")):
            if os.path.exists(cand):
                vp = pd.read_csv(cand)[["Dicom_id", "ViewPosition"]]
                bank = bank.merge(vp, on="Dicom_id", how="left")
                miss = int(bank.ViewPosition.isna().sum())
                print(f"  已合并 ViewPosition（缺失 {miss} 张）来源={os.path.relpath(cand, ROOT)}")
                break
    print(f"  概率库: {len(bank):,} 张图 / {bank.Study_id.nunique():,} 个检查")
    if "ViewPosition" not in bank.columns:
        print("  [warn] 仍无 ViewPosition 列，侧位权重无法生效")

    if a.check:
        ref = pd.read_csv(SHIPPED)
        # 先试无侧位（因为库里可能没有视角列）
        lat = 1.0 if "ViewPosition" not in bank.columns else 0.6
        got = build(bank, lat, 0.85)
        print(f"  重建行数 {len(got):,}  vs 包内提交 {len(ref):,}")
        same_keys = got[["Subject_id", "Study_id", "Predict_class"]].equals(
            ref[["Subject_id", "Study_id", "Predict_class"]])
        dmax = float(np.abs(got["Probability"].values - ref["Probability"].values).max())
        print(f"  键序列一致: {'是' if same_keys else '否'}")
        print(f"  概率最大差: {dmax:.3e}")
        print(f"  逐位相同: {'✅ 是' if (same_keys and dmax == 0) else '❌ 否'}")
        print(f"  包内提交 sha: {sha256(SHIPPED)[:32]}")
        return

    sbc = {}
    if a.shrink_by_class:
        for part in a.shrink_by_class.split(","):
            k, v = part.split(":")
            sbc[str(int(k))] = float(v)
    out = build(bank, a.lat_weight, a.shrink, sbc)
    tag = a.tag or f"shrink{a.shrink}_lat{a.lat_weight}"
    os.makedirs(OUTDIR, exist_ok=True)
    p = os.path.join(OUTDIR, f"{tag}.csv")
    out.to_csv(p, index=False, encoding="utf-8")
    print(f"  [wrote] {os.path.relpath(p, ROOT)}  rows={len(out):,}  sha={sha256(p)[:24]}")


if __name__ == "__main__":
    main()
