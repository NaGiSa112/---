"""提交文件自检（每次上传前跑）。

检查项：
  1. 列名与顺序与包内 submission.csv 完全一致
  2. 行数 = 检查数 × 10
  3. 键集与官方 test.csv 完全一致（不多不少、无重复）
  4. 概率全部在 (0, 1)、无 NaN
  5. 每检查恰好 10 类
  6. 与参考文件的键序列一致（若给 --ref）

    python tools/check_submission.py submissions/ladder_2026-11-23/V1_class2_045.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = r"E:\挑战赛"
COLS = ["Subject_id", "Study_id", "Predict_class", "Probability"]

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--ref", default=r"E:\挑战赛\_pkgcheck\陈思阳_最终提交材料\model\submission.csv")
    ap.add_argument("--test-csv", default=r"E:\挑战赛\cxr_first\test.csv")
    a = ap.parse_args()

    d = pd.read_csv(a.path)
    fails = []

    ok = list(d.columns) == COLS
    print(f"  [{'OK ' if ok else 'FAIL'}] 列名与顺序 {list(d.columns)}")
    if not ok:
        fails.append("columns")

    test = pd.read_csv(a.test_csv)
    exp = test[["Subject_id", "Study_id"]].drop_duplicates()
    n_study = len(exp)
    ok = len(d) == n_study * 10
    print(f"  [{'OK ' if ok else 'FAIL'}] 行数 {len(d):,} = 检查数 {n_study:,} × 10")
    if not ok:
        fails.append("rows")

    got = set(map(tuple, d[["Subject_id", "Study_id", "Predict_class"]].values))
    want = set()
    for s, st in exp.values:
        for c in range(1, 11):
            want.add((s, st, c))
    ok = got == want
    print(f"  [{'OK ' if ok else 'FAIL'}] 键集完全一致（缺失 {len(want - got)}、多余 {len(got - want)}）")
    if not ok:
        fails.append("keys")

    ok = not d.duplicated(["Subject_id", "Study_id", "Predict_class"]).any()
    print(f"  [{'OK ' if ok else 'FAIL'}] 无重复键")
    if not ok:
        fails.append("dup")

    p = d["Probability"].to_numpy()
    ok = bool(np.isfinite(p).all() and (p > 0).all() and (p < 1).all())
    print(f"  [{'OK ' if ok else 'FAIL'}] 概率在 (0,1) 且无 NaN（min={p.min():.6f}, max={p.max():.6f}）")
    if not ok:
        fails.append("prob")

    cnt = d.groupby(["Subject_id", "Study_id"]).size()
    ok = bool((cnt == 10).all())
    print(f"  [{'OK ' if ok else 'FAIL'}] 每检查恰好 10 类（异常 {int((cnt != 10).sum())} 个）")
    if not ok:
        fails.append("per_study")

    if os.path.exists(a.ref):
        r = pd.read_csv(a.ref)
        same = d[["Subject_id", "Study_id", "Predict_class"]].equals(
            r[["Subject_id", "Study_id", "Predict_class"]])
        print(f"  [{'OK ' if same else 'FAIL'}] 键序列与包内提交一致")
        if not same:
            fails.append("ref_order")
        dp = float(np.abs(p - r["Probability"].to_numpy()).max())
        print(f"       与包内提交的最大概率差 = {dp:.3e}")

    print()
    if fails:
        print(f"  结论：❌ 未通过（{fails}）")
        sys.exit(1)
    print("  结论：✅ 全部通过，可以上传")


if __name__ == "__main__":
    main()
