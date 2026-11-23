"""预测完整性检查：发现任何问题就失败，不允许静默取交集。

审阅（2026-11-20 §3.1）指出 `compare_folds.py` 用"取第一条、去重、取交集"的写法，
可能把漏预测或重复预测藏起来。本工具按冻结折表构造**精确预期键集**，逐项核对：

  1. 每个折的 study CSV 必须恰好覆盖该折全部 study，不多不少
  2. 每个 study 必须恰好有 1—10 十个类别各一行
  3. 不允许重复键、缺失键、额外键、NaN、超出 [0,1] 的概率
  4. 患者不得跨折（折表自身的一致性）
  5. 图级 CSV（若存在）必须覆盖该折全部 Dicom_id

用法：
    python tools/verify_predictions.py --tag blend48 --folds 0,1,2,3,4
    python tools/verify_predictions.py --tag init22k --folds 0,1
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd

ROOT = r"E:\挑战赛"
FOLDS = os.path.join(ROOT, "04_新一轮", "00_规则", "patient_folds_5f.csv")
PCOLS = list(range(1, 11))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_fold(tag: str, f: int, fold_tab: pd.DataFrame, train: pd.DataFrame) -> list[str]:
    """返回问题列表；空列表表示通过。"""
    problems: list[str] = []
    expect = set(fold_tab.loc[fold_tab.fold == f, "Study_id"].astype(str))
    if not expect:
        return [f"fold {f}: 折表里没有 study"]

    study_csv = os.path.join(ROOT, "runs", f"{tag}_f{f}_study.csv")
    if not os.path.exists(study_csv):
        return [f"fold {f}: 缺少 {os.path.basename(study_csv)}"]

    d = pd.read_csv(study_csv)
    for col in ("Subject_id", "Study_id", "Predict_class", "Probability"):
        if col not in d.columns:
            return [f"fold {f}: 缺少列 {col}"]

    got = set(d.Study_id.astype(str))
    if got - expect:
        problems.append(f"fold {f}: 多出 {len(got - expect)} 个不属于该折的 study")
    if expect - got:
        problems.append(f"fold {f}: 缺少 {len(expect - got)} 个该折的 study")

    dup = d.duplicated(subset=["Subject_id", "Study_id", "Predict_class"]).sum()
    if dup:
        problems.append(f"fold {f}: {dup} 个重复键 (Subject_id, Study_id, Predict_class)")

    per_study = d.groupby(["Subject_id", "Study_id"]).Predict_class.nunique()
    if (per_study != 10).any():
        problems.append(f"fold {f}: {int((per_study != 10).sum())} 个 study 的类别数不等于 10")

    cls = set(d.Predict_class.astype(int).unique())
    if cls != set(PCOLS):
        problems.append(f"fold {f}: 类别集为 {sorted(cls)}，不是 1—10")

    p = pd.to_numeric(d.Probability, errors="coerce")
    if p.isna().any():
        problems.append(f"fold {f}: {int(p.isna().sum())} 个概率为 NaN/非数值")
    valid = p.dropna()
    if ((valid < 0) | (valid > 1)).any():
        problems.append(f"fold {f}: {int(((valid < 0) | (valid > 1)).sum())} 个概率超出 [0,1]")

    # 图级 CSV（可选）
    img_csv = os.path.join(ROOT, "runs", f"{tag}_f{f}_image.csv")
    if os.path.exists(img_csv):
        di = pd.read_csv(img_csv)
        if "Dicom_id" in di.columns:
            want = set(train.loc[train.Study_id.astype(str).isin(expect), "Dicom_id"].astype(str))
            have = set(di.Dicom_id.astype(str))
            if want - have:
                problems.append(f"fold {f}: 图级预测缺少 {len(want - have)} 张图")
            if have - want:
                problems.append(f"fold {f}: 图级预测多出 {len(have - want)} 张图")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--fold-csv", default=FOLDS)
    a = ap.parse_args()
    folds = [int(x) for x in a.folds.split(",") if x.strip()]

    fold_tab = pd.read_csv(a.fold_csv, usecols=["Study_id", "fold"])
    fold_tab["Study_id"] = fold_tab.Study_id.astype(str)
    train = pd.read_csv(os.path.join(ROOT, "cxr_first", "train.csv"))

    print(f"  折表 {os.path.relpath(a.fold_csv, ROOT)}  sha256 {sha256(a.fold_csv)[:16]}…")
    print(f"  待检查 tag={a.tag} 折={folds}")

    # 折表自身：患者不得跨折
    per_pat = train.assign(Study_id=train.Study_id.astype(str)).merge(
        fold_tab, on="Study_id", how="inner").groupby("Subject_id").fold.nunique()
    if (per_pat > 1).any():
        print(f"  ❌ 折表本身有 {int((per_pat > 1).sum())} 名患者跨折")
        raise SystemExit(1)

    all_problems: list[str] = []
    for f in folds:
        probs = check_fold(a.tag, f, fold_tab, train)
        csv = os.path.join(ROOT, "runs", f"{a.tag}_f{f}_study.csv")
        h = sha256(csv)[:16] + "…" if os.path.exists(csv) else "—"
        if probs:
            print(f"  ❌ fold {f}: {len(probs)} 项问题")
            for p in probs:
                print(f"     - {p}")
        else:
            n = len(fold_tab[fold_tab.fold == f])
            print(f"  ✅ fold {f}: {n} study / 10 类齐全 / 无重复无缺失  sha {h}")
        all_problems += probs

    print()
    if all_problems:
        print(f"  ❌ 共 {len(all_problems)} 项问题，**拒绝用于比较**")
        raise SystemExit(1)
    if len(folds) < 5:
        print(f"  ⚠️ 只检查了 {len(folds)} 折（{folds}）：报告里只能称『部分折』，不能称完整 OOF")
    print(f"  ✅ 全部通过（{len(folds)} 折）")


if __name__ == "__main__":
    main()
