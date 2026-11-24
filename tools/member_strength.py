"""现有成员强度画像（2026-11-24）。

回答四件事：
  1. 每个成员**单独**在 valid 上的检查级宏 AUC 与逐类 AUC
  2. 成员之间的排序相关性（多样性）
  3. 集成后 vs 去掉某个成员（leave-one-out）
  4. 供体权重扫描（0.00 ~ 0.30）与 Tiny 等权的对比

做法：用**交付包内**的配置逐成员各跑一次 valid（约 90 秒/成员），再离线聚合。

    python tools/member_strength.py --predict   # 先跑 4 次推理
    python tools/member_strength.py             # 只做分析
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = r"E:\挑战赛"
PKG = os.path.join(ROOT, "_pkgcheck", "陈思阳_最终提交材料")
DEPLOY = os.path.join(PKG, "model", "deploy.json")
OUTDIR = os.path.join(ROOT, "runs", "member_valid")
PCOLS = list(range(1, 11))
REPORT = os.path.join(ROOT, "reports", "成员强度_2026-11-24.md")

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
    o = np.argsort(p)
    r = np.empty(len(p), float)
    r[o] = np.arange(1, len(p) + 1)
    pos = y > 0
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def aggregate(df: pd.DataFrame, lat_weight: float = 0.6, shrink: float = 0.85) -> pd.DataFrame:
    d = df.copy()
    if "ViewPosition" in d.columns:
        w = np.where(np.isin(d["ViewPosition"].fillna("NA").values, ["LATERAL", "LL"]),
                     lat_weight, 1.0)
    else:
        w = np.ones(len(d))
    cols = [f"p{k}" for k in range(1, 11)]
    for c in cols:
        d[c] = d[c] * w
    g = d.groupby(["Subject_id", "Study_id"], as_index=False)[cols].sum()
    ws = pd.Series(w).groupby([d["Subject_id"].values, d["Study_id"].values]).sum()
    g[cols] = g[cols].div(ws.values[:, None])
    if shrink < 1.0:
        pm = g.groupby("Subject_id")[cols].transform("mean").values
        g[cols] = shrink * g[cols].values + (1 - shrink) * pm
    g.columns = ["Subject_id", "Study_id"] + PCOLS
    return g.set_index(["Subject_id", "Study_id"])


def predict_members() -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    spec = json.load(open(DEPLOY, encoding="utf-8"))
    for m in spec["members"]:
        out = os.path.join(OUTDIR, f"{m['name']}.csv")
        if os.path.exists(out):
            print(f"   {m['name']}: 已有，跳过")
            continue
        mm = dict(m, weight=1.0)
        ck = mm.get("ckpt", "")
        if ck and not os.path.isabs(ck):
            mm["ckpt"] = os.path.join(PKG, "model", os.path.basename(ck))
        single = {"amp": spec["amp"], "fusion": "arithmetic",
                  "cpu_threads": spec.get("cpu_threads", 4),
                  "graph_tiny": spec.get("graph_tiny", True), "members": [mm]}
        sp = os.path.join(OUTDIR, f"_spec_{m['name']}.json")
        json.dump(single, open(sp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        cmd = [sys.executable, os.path.join(ROOT, "src", "predict.py"), "--split", "valid",
               "--batch-size", "1", "--deploy", sp,
               "--out", os.path.join(OUTDIR, f"_sub_{m['name']}.csv"), "--probs-out", out]
        env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, ".pylibs") + os.pathsep
                   + os.path.join(ROOT, "src"), PYTHONIOENCODING="utf-8",
                   CXR_DATA=os.path.join(ROOT, "cxr_first"),
                   TORCH_HOME=os.path.join(ROOT, ".torch_home"))
        r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        print(f"   {m['name']}: rc={r.returncode}")
        if r.returncode != 0:
            print("     ", (r.stderr or "")[-300:])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict", action="store_true")
    a = ap.parse_args()
    if a.predict:
        predict_members()

    spec = json.load(open(DEPLOY, encoding="utf-8"))
    names = [m["name"] for m in spec["members"]]
    w0 = {m["name"]: float(m["weight"]) for m in spec["members"]}

    va = pd.read_csv(os.path.join(ROOT, "cxr_first", "valid.csv"))
    Y = np.stack([parse_label(s) for s in va.Predict_class])
    lab = pd.DataFrame(Y, columns=PCOLS)
    lab["Subject_id"] = va.Subject_id.values
    lab["Study_id"] = va.Study_id.values
    lab = lab.groupby(["Subject_id", "Study_id"])[PCOLS].max()

    vp = va[["Dicom_id", "ViewPosition"]]
    per_member = {}
    for n in names:
        p = os.path.join(OUTDIR, f"{n}.csv")
        d = pd.read_csv(p).merge(vp, on="Dicom_id", how="left")
        per_member[n] = aggregate(d)
    idx = lab.index
    for n in names:
        per_member[n] = per_member[n].loc[idx]

    # ---- 1) 单成员强度 ----
    rows = []
    for n in names:
        per = [auc(lab[c].to_numpy(), per_member[n][c].to_numpy()) for c in PCOLS]
        rows.append((n, w0[n], float(np.nanmean(per)), per))
    rows.sort(key=lambda r: -r[2])

    # ---- 2) 相关性 ----
    corr = pd.DataFrame(index=names, columns=names, dtype=float)
    for i in names:
        for j in names:
            corr.loc[i, j] = float(pd.Series(per_member[i][1]).corr(
                pd.Series(per_member[j][1]), method="spearman"))

    # ---- 3) 集成与留一 ----
    def fuse(weights: dict) -> pd.DataFrame:
        acc = np.zeros((len(idx), 10))
        acc_sq = np.zeros((len(idx), 10))
        ws = 0.0
        for n, w in weights.items():
            p = per_member[n].to_numpy(np.float64)
            acc += w * p
            acc_sq += w * np.square(p)
            ws += w
        out = 0.5 * (acc / ws) + 0.5 * np.sqrt(acc_sq / ws)
        return pd.DataFrame(out, index=idx, columns=PCOLS)

    def macro(df: pd.DataFrame) -> float:
        return float(np.nanmean([auc(lab[c].to_numpy(), df[c].to_numpy()) for c in PCOLS]))

    full = macro(fuse(w0))
    loo = {}
    for n in names:
        w = {k: v for k, v in w0.items() if k != n}
        if w:
            loo[n] = macro(fuse(w)) - full
    tiny_only = {k: v for k, v in w0.items() if k != "donor_v21"}
    tiny_auc = macro(fuse(tiny_only))
    equal_tiny = {k: 1.0 for k in names if k != "donor_v21"}
    equal_tiny_auc = macro(fuse(equal_tiny))

    # ---- 4) 供体权重扫描 ----
    sweep = []
    for wd in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
        w = {k: (1.0 - wd) / 3.0 for k in names if k != "donor_v21"}
        if wd > 0:
            w["donor_v21"] = wd
        sweep.append((wd, macro(fuse(w))))

    # ---- 报告 ----
    L = ["# 现有成员强度画像（2026-11-24）", "",
         f"- valid：{len(idx):,} 个检查；权重取自交付包 `model/deploy.json`",
         "- 聚合口径与交付一致：侧位 0.6 → 检查内加权均值 → 患者收缩 0.85", "",
         "## 1　单成员强度（单独跑，不集成）", "",
         "| 成员 | 交付权重 | 单独宏 AUC | 类1 | 类2 | 类4 | 类8 |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for n, w, m, per in rows:
        L.append(f"| `{n}` | {w:.4f} | **{m:.4f}** | {per[0]:.3f} | {per[1]:.3f} | "
                 f"{per[3]:.3f} | {per[7]:.3f} |")
    L += ["", "## 2　成员间排序相关性（Spearman，检查级）", "",
          "| | " + " | ".join(n[:14] for n in names) + " |",
          "|---|" + "---:|" * len(names)]
    for i in names:
        L.append(f"| `{i[:18]}` | " + " | ".join(f"{corr.loc[i, j]:.3f}" for j in names) + " |")
    L += ["", "## 3　集成与留一", "",
          f"- 交付四成员集成（half-power，权重 0.2833×3 + 0.15）：宏 AUC = **{full:.6f}**",
          f"- 只留三个 Tiny（各 1/3）：{tiny_auc:.6f}（{tiny_auc - full:+.6f}）",
          f"- 三个 Tiny 等权（不归一）：{equal_tiny_auc:.6f}", "",
          "| 去掉谁 | 宏 AUC 变化 |", "|---|---:|"]
    for n in names:
        L.append(f"| `{n}` | {loo[n]:+.6f} |")
    L += ["", "## 4　供体权重扫描", "", "| 供体权重 | 宏 AUC | 相对现值 |", "|---:|---:|---:|"]
    for wd, m in sweep:
        L.append(f"| {wd:.2f} | {m:.6f} | {m - full:+.6f} |")
    best = max(sweep, key=lambda r: r[1])
    L += ["", f"扫描最优：供体 {best[0]:.2f}（{best[1]:.6f}），"
              f"比现值 0.15 高 **{best[1] - full:+.6f}**。"]

    open(REPORT, "w", encoding="utf-8").write("\n".join(L))
    json.dump({"members": [{"name": n, "weight": w, "macro": m, "per_class": per}
                           for n, w, m, per in rows],
               "corr": corr.to_dict(), "ensemble": full, "loo": loo,
               "tiny_only": tiny_auc, "donor_sweep": sweep},
              open(os.path.join(ROOT, "reports", "member_strength.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n".join(L))
    print(f"\n  [wrote] {os.path.relpath(REPORT, ROOT)}")


if __name__ == "__main__":
    main()
