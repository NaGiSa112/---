"""对比两份患者折表，回答两个问题（用于跨机折表一致性核对）。

背景：折号不同 ⇒ 训练集可能包含对方 fold 2 的检查，对方 fold 2 上的评分就是泄漏后的虚高。
每折规模相同**不能**证明划分相同（不同划分可以有相同的折大小），必须逐 Study 比对键集。

    python tools/compare_fold_tables.py --a data/patient_folds_5f.csv --b 别人给的折表.csv

输出（正是需要的两行）：
    折号不一致的 Study：N
    若以 A 的 fold 2 为评估集：B 的训练集里有 M 个 A-fold2 检查
另附反向一行，便于双向确认。

注意：A/B 两份文件的**行尾符差异不影响结论**（pandas 读入时已归一），但字节 SHA 会不同，
所以"SHA 不一致"必须用本脚本排除划分差异后才能下结论。
"""
from __future__ import annotations

import argparse
import hashlib
import os

import pandas as pd

COLS = ("Study_id", "Subject_id", "fold")


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"Study_id": str, "Subject_id": str})
    miss = [c for c in COLS if c not in df.columns]
    if miss:
        raise SystemExit(f"[error] {path} 缺少列 {miss}（需要 Study_id, Subject_id, fold）")
    return df[list(COLS)].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data", "patient_folds_5f.csv"), help="基准折表 A")
    ap.add_argument("--b", required=True, help="待核对的折表 B（你们的）")
    ap.add_argument("--fold", type=int, default=2, help="评估折号，默认 2")
    a = ap.parse_args()

    A, B = load(a.a), load(a.b)
    for name, df, p in (("A", A, a.a), ("B", B, a.b)):
        dup = len(df) - df.Study_id.nunique()
        print(f"[load] {name} {p}: {len(df)} studies, 重复 {dup}, "
              f"patients {df.Subject_id.nunique()}, sha256 {hashlib.sha256(open(p, 'rb').read()).hexdigest()}")

    # ---- 折号不一致 ----
    m = A.merge(B, on="Study_id", how="outer", suffixes=("_a", "_b"), indicator=True)
    only_a = m[m._merge == "left_only"]
    only_b = m[m._merge == "right_only"]
    both = m[m._merge == "both"]
    fold_diff = both[both.fold_a != both.fold_b]
    pat_diff = both[both.Subject_id_a.astype(str) != both.Subject_id_b.astype(str)]
    print(f"\n折号不一致的 Study：{len(fold_diff)}")
    if len(only_a) or len(only_b):
        print(f"  另：只在 A 有的 Study {len(only_a)} 个，只在 B 有的 {len(only_b)} 个")
    if len(pat_diff):
        print(f"  另：同一 Study 的患者归属不一致 {len(pat_diff)} 个（更严重，需先查）")
    if len(fold_diff):
        print("  例：" + ", ".join(f"{r.Study_id}(A{r.fold_a}→B{r.fold_b})"
                                   for r in fold_diff.head(5).itertuples()))

    # ---- 双向泄漏 ----
    a_eval = set(A[A.fold == a.fold].Study_id)
    b_eval = set(B[B.fold == a.fold].Study_id)
    b_train = set(B[B.fold != a.fold].Study_id)
    a_train = set(A[A.fold != a.fold].Study_id)
    print(f"若以 A 的 fold {a.fold} 为评估集：B 的训练集里有 {len(a_eval & b_train)} 个 A-fold{a.fold} 检查")
    print(f"若以 B 的 fold {a.fold} 为评估集：A 的训练集里有 {len(b_eval & a_train)} 个 B-fold{a.fold} 检查")

    ok = (len(fold_diff) == 0 and not len(only_a) and not len(only_b)
          and not len(pat_diff) and not (a_eval & b_train))
    print("\n[结论]", "两份折表划分一致，无泄漏风险" if ok else "两份折表划分不同，评分不可直接比较")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
