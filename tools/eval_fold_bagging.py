"""折模型 Bagging 评估（零训练，GPU 只需一次推理）。

想法
----
`runs/oof_f{0..4}/last.pt` 是 5 个 Tiny，每个只见过 train 的 80%，且**见过的 80% 各不相同**。
把它们当成员做 bagging，误差结构会比"同一份全量数据训练的三个种子"更多样。
我们此前测过的"六模型集成"之所以零增益，是因为那 6 个是在**同一份全量数据**上训练的，
在本机同配方下预测逐位相同（见受控重复实验）。

代价
----
体积：5×111.36 + 43.37 = 600 MB，超 500 MB 上限 ⇒ 最多 4 个折模型 + 供体（489.8 MB）。
延迟：每个 Tiny 约 12–14 ms（CUDA Graph），4 个约 55–60 ms，加供体约 70 ms，仍在 100 ms 内。

用法
----
    python tools/eval_fold_bagging.py --cache-only     # 只看已有预测，能算的组合直接算
    python tools/eval_fold_bagging.py --predict        # 用 GPU 为缺失的折模型生成 valid 预测
    python tools/eval_fold_bagging.py --predict --with-donor

输出
----
  runs/foldbag_valid.csv（每个折模型一列，宽表）
  reports/折bagging_2026-11-21.md
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = r"E:\挑战赛"
PCOLS = list(range(1, 11))
FOLDS = os.path.join(ROOT, "04_新一轮", "00_规则", "patient_folds_5f.csv")

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


def wide(path: str, col_prefix: str = "") -> pd.DataFrame:
    d = pd.read_csv(path)
    df = d.pivot_table(index=["Subject_id", "Study_id"], columns="Predict_class",
                       values="Probability", aggfunc="first").reindex(columns=PCOLS)
    if col_prefix:
        df.columns = [f"{col_prefix}{c}" for c in df.columns]
    return df


def make_deploy(members: list[dict], out_path: str) -> None:
    spec = {"amp": "bf16", "members": members, "fusion": "half_power",
            "cpu_threads": 4, "graph_tiny": True}
    json.dump(spec, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict", action="store_true", help="用 GPU 生成缺失的折模型 valid 预测")
    ap.add_argument("--per-fold-timeout", type=int, default=900)
    a = ap.parse_args()

    cache = os.path.join(ROOT, "runs", "foldbag_valid.csv")
    have = os.path.exists(cache)
    if a.predict and not have:
        # 逐折模型单独预测 valid（每个约 90 秒）
        rows = []
        for f in range(5):
            ck = os.path.join(ROOT, "runs", f"oof_f{f}", "last.pt")
            if not os.path.exists(ck):
                print(f"  跳过 fold {f}：无权重")
                continue
            spec = os.path.join(ROOT, "model", f"_tmp_foldbag_f{f}.json")
            make_deploy([{"name": f"oof_f{f}", "ckpt": f"runs/oof_f{f}/last.pt",
                          "img_size": 544, "cache_size": 512, "norm": "per_image",
                          "tta": True, "weight": 1.0, "lse_alpha": 0.3}], spec)
            out = os.path.join(ROOT, "runs", f"foldbag_f{f}_valid.csv")
            print(f"  预测 fold {f} …", flush=True)
            rc = subprocess.run(
                [sys.executable, os.path.join(ROOT, "src", "predict.py"),
                 "--split", "valid", "--batch-size", "1", "--lat-weight", "0.6",
                 "--shrink", "1.0", "--deploy", spec, "--out", out],
                cwd=ROOT, env=dict(os.environ, PYTHONPATH=os.path.join(ROOT, ".pylibs")
                                   + os.pathsep + os.path.join(ROOT, "src"),
                                   PYTHONIOENCODING="utf-8"),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                encoding="utf-8", errors="replace", timeout=a.per_fold_timeout).returncode
            print(f"    rc={rc}")
            if rc == 0 and os.path.exists(out):
                rows.append(wide(out))
        if rows:
            df = pd.concat(rows, axis=1)
            df.to_csv(cache)
            print(f"  [wrote] {os.path.relpath(cache, ROOT)}  列数 {df.shape[1]}")

    # ---- 评估：只报"未收缩"到"a=0.85"的对比，统一口径 ----
    va = pd.read_csv(os.path.join(ROOT, "cxr_first", "valid.csv"))
    Y = np.stack([parse_label(s) for s in va.Predict_class])
    lab = pd.DataFrame(Y, columns=PCOLS)
    lab["Subject_id"] = va.Subject_id.values
    lab["Study_id"] = va.Study_id.values
    lab = lab.groupby(["Subject_id", "Study_id"])[PCOLS].max()

    results = []
    # 基线：交付版四成员
    base = pd.read_csv(os.path.join(ROOT, "runs", "orig_valid.csv"))
    bw = base.pivot_table(index=["Subject_id", "Study_id"], columns="Predict_class",
                          values="Probability", aggfunc="first").reindex(columns=PCOLS)
    idx = bw.index.intersection(lab.index)
    per = [auc(lab.loc[idx, c].to_numpy(), bw.loc[idx, c].to_numpy()) for c in PCOLS]
    results.append(("基线：3 Tiny + 供体（交付版）", float(np.nanmean(per)), per))

    if have:
        fb = pd.read_csv(cache, index_col=[0, 1])
        fb.index.names = ["Subject_id", "Study_id"]
        # 每个折模型单独
        n_models = fb.shape[1] // 10
        for k in range(n_models):
            sub = fb.iloc[:, k * 10:(k + 1) * 10]
            sub.columns = PCOLS
            idx2 = sub.index.intersection(lab.index)
            per2 = [auc(lab.loc[idx2, c].to_numpy(), sub.loc[idx2, c].to_numpy()) for c in PCOLS]
            results.append((f"单个折模型 f{k}", float(np.nanmean(per2)), per2))
        # 组合
        for r in range(2, n_models + 1):
            for combo in itertools.combinations(range(n_models), r):
                cols = []
                for k in combo:
                    cols += list(range(k * 10, (k + 1) * 10))
                sub = fb.iloc[:, cols]
                arr = np.mean([sub.iloc[:, k * 10:(k + 1) * 10].to_numpy()
                               for k in range(len(combo))], axis=0)
                sub2 = pd.DataFrame(arr, index=sub.index, columns=PCOLS)
                idx3 = sub2.index.intersection(lab.index)
                per3 = [auc(lab.loc[idx3, c].to_numpy(), sub2.loc[idx3, c].to_numpy())
                        for c in PCOLS]
                results.append((f"bagging {len(combo)} 个折模型 {combo}",
                                float(np.nanmean(per3)), per3))

    results.sort(key=lambda r: -r[1])
    L = ["# 折模型 bagging 评估（2026-11-21）", "",
         "> 折模型各自只见过 train 的 80%，且子集不同 ⇒ 误差结构比同数据多种子更多样。",
         "> 体积约束：最多 4 个折模型 + 供体（489.8 MB）。", "",
         "| 配置 | 检查级宏 AUC | 相对基线 |", "|---|---:|---:|"]
    base_m = results[[i for i, r in enumerate(results) if r[0].startswith("基线")][0]][1] \
        if any(r[0].startswith("基线") for r in results) else float("nan")
    for name, m, _ in results[:30]:
        L.append(f"| {name} | {m:.6f} | {m - base_m:+.6f} |")
    if not have:
        L += ["", "> 尚未生成折模型的 valid 预测。用 `--predict` 先跑一次（GPU 约 7 分钟）。"]
    out_md = os.path.join(ROOT, "reports", "折bagging_2026-11-21.md")
    open(out_md, "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L[:24]))
    print(f"\n  [wrote] {os.path.relpath(out_md, ROOT)}")


if __name__ == "__main__":
    main()
