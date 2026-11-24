"""E1+E2 自动判读：给一个新的候选成员，直接回答"够不够格、值不值得进集成"。

输入
----
`--probs` 候选成员在 valid 上的逐图概率 CSV（列：Dicom_id, Subject_id, Study_id, p1..p10，
由 `src/predict.py --probs-out` 产生）。

判据（与方案一致）
------------------
E1 单模：检查级宏 AUC ≥ 0.8200
E2 多样性：与三个 Tiny 的平均 Spearman < 0.90
E2 集成：最优权重下 Δ 宏 AUC ≥ +0.0015

    python tools/e1_score.py --probs runs/candidate_f2_valid.csv --name caformer
    python tools/e1_score.py --probs runs/member_valid/donor_v21.csv --name donor  # 自检
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

# 路径全部走环境变量，云端与本地通用（曾经硬编码 E:\挑战赛 导致云端 FileNotFoundError）
ROOT = os.environ.get("CXR_ROOT_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
try:
    from cxr.config import DATA as _DATA
except Exception:                                       # noqa: BLE001
    _DATA = os.path.join(ROOT, "cxr_first")
MEMBER_DIR = os.environ.get("CXR_MEMBER_DIR") or os.path.join(ROOT, "runs", "member_valid")
LABEL_CSV = os.path.join(_DATA, "valid.csv")
PCOLS = list(range(1, 11))
OUT_MD = os.path.join(ROOT, "reports", "E1E2_判读_{name}_2026-11-24.md")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

GATE_MACRO = 0.8200
GATE_SPEAR = 0.90
GATE_DELTA = 0.0015
TINY = ["cnxt512_pat_e12", "cnxt512_pat_s43_e12", "tiny_seed44_last12"]
DONOR = "donor_v21"


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


def aggregate(df: pd.DataFrame, lat: float = 0.6, shrink: float = 0.85) -> pd.DataFrame:
    d = df.copy()
    if "ViewPosition" in d.columns:
        w = np.where(np.isin(d["ViewPosition"].fillna("NA").values, ["LATERAL", "LL"]), lat, 1.0)
    else:
        w = np.ones(len(d))
    cols = [f"p{k}" for k in range(1, 11)]
    for c in cols:
        d[c] = d[c] * w
    g = d.groupby(["Subject_id", "Study_id"], as_index=False)[cols].sum()
    ws = pd.Series(w).groupby([d["Subject_id"].values, d["Study_id"].values]).sum()
    g[cols] = g[cols].div(ws.values[:, None])
    pm = g.groupby("Subject_id")[cols].transform("mean").values
    g[cols] = shrink * g[cols].values + (1 - shrink) * pm
    g.columns = ["Subject_id", "Study_id"] + PCOLS
    return g.set_index(["Subject_id", "Study_id"])


def load_member(path: str, vp: pd.DataFrame) -> pd.DataFrame:
    d = pd.read_csv(path)
    if "ViewPosition" not in d.columns and "Dicom_id" in d.columns:
        d = d.merge(vp, on="Dicom_id", how="left")
    return aggregate(d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True)
    ap.add_argument("--name", default="candidate")
    a = ap.parse_args()

    va = pd.read_csv(LABEL_CSV)
    Y = np.stack([parse_label(s) for s in va.Predict_class])
    lab = pd.DataFrame(Y, columns=PCOLS)
    lab["Subject_id"] = va.Subject_id.values
    lab["Study_id"] = va.Study_id.values
    lab = lab.groupby(["Subject_id", "Study_id"])[PCOLS].max()
    vp = va[["Dicom_id", "ViewPosition"]]

    cand = load_member(a.probs, vp)
    members = {}
    for n in TINY + [DONOR]:
        p = os.path.join(MEMBER_DIR, f"{n}.csv")
        if os.path.exists(p):
            members[n] = load_member(p, vp)
    if not members:
        raise SystemExit("[error] 找不到现有成员的 valid 预测，先跑 tools/member_strength.py --predict")

    idx = lab.index
    cand = cand.loc[idx]
    for n in members:
        members[n] = members[n].loc[idx]

    def macro(df: pd.DataFrame) -> float:
        return float(np.nanmean([auc(lab[c].to_numpy(), df[c].to_numpy()) for c in PCOLS]))

    per = [float(np.nanmean([auc(lab[c].to_numpy(), cand[c].to_numpy())])) for c in PCOLS]
    cand_macro = float(np.nanmean(per))

    spears = {}
    for n in TINY:
        if n in members:
            spears[n] = float(pd.Series(cand[1]).corr(pd.Series(members[n][1]), method="spearman"))
    mean_spear = float(np.mean(list(spears.values()))) if spears else float("nan")

    # 现集成（权重取自交付）
    base_w = {n: 0.2833333333333333 for n in TINY if n in members}
    base_w[DONOR] = 0.15
    base_w = {k: v for k, v in base_w.items() if k in members}

    # 若候选本身就是某个已在集成里的成员（自检情形），把它从基线里剔除，
    # 否则"添加"会把同一成员计两次，得到无意义的负增益。
    self_member = None
    for n, m in members.items():
        if m.shape == cand.shape and np.allclose(m.to_numpy(), cand.to_numpy(),
                                                 atol=1e-9, rtol=0):
            self_member = n
            break
    if self_member:
        base_w.pop(self_member, None)
        print(f"  [检测] 候选与现有成员 `{self_member}` 数值相同，已从基线中剔除该成员")

    def fuse(weights: dict) -> pd.DataFrame:
        acc = np.zeros((len(idx), 10))
        acc_sq = np.zeros((len(idx), 10))
        ws = 0.0
        for n, w in weights.items():
            p = members[n].to_numpy(np.float64) if n in members else cand.to_numpy(np.float64)
            acc += w * p
            acc_sq += w * np.square(p)
            ws += w
        out = 0.5 * (acc / ws) + 0.5 * np.sqrt(acc_sq / ws)
        return pd.DataFrame(out, index=idx, columns=PCOLS)

    base = macro(fuse(base_w))

    # 添加：候选按不同权重并入
    add = []
    for w in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        wts = {k: v * (1 - w) for k, v in base_w.items()}
        acc = np.zeros((len(idx), 10))
        acc_sq = np.zeros((len(idx), 10))
        ws = 0.0
        for n, ww in wts.items():
            p = members[n].to_numpy(np.float64)
            acc += ww * p
            acc_sq += ww * np.square(p)
            ws += ww
        p = cand.to_numpy(np.float64)
        acc += w * p
        acc_sq += w * np.square(p)
        ws += w
        out = 0.5 * (acc / ws) + 0.5 * np.sqrt(acc_sq / ws)
        add.append((w, macro(pd.DataFrame(out, index=idx, columns=PCOLS)) - base))

    # 替换：去掉一个 Tiny，候选补位
    repl = []
    for drop in TINY:
        if drop not in members:
            continue
        wts = {k: v for k, v in base_w.items() if k != drop}
        tot = sum(wts.values())
        wts = {k: v / tot for k, v in wts.items()}
        acc = np.zeros((len(idx), 10))
        acc_sq = np.zeros((len(idx), 10))
        ws = 0.0
        for n, ww in wts.items():
            p = members[n].to_numpy(np.float64)
            acc += ww * p
            acc_sq += ww * np.square(p)
            ws += ww
        p = cand.to_numpy(np.float64)
        acc += 1.0 * p
        acc_sq += np.square(p)
        ws += 1.0
        out = 0.5 * (acc / ws) + 0.5 * np.sqrt(acc_sq / ws)
        repl.append((drop, macro(pd.DataFrame(out, index=idx, columns=PCOLS)) - base))

    best_add = max(add, key=lambda r: r[1])
    best_repl = max(repl, key=lambda r: r[1]) if repl else (None, -9)

    g1 = cand_macro >= GATE_MACRO
    g2 = mean_spear < GATE_SPEAR
    g3 = max(best_add[1], best_repl[1]) >= GATE_DELTA

    L = [f"# E1/E2 判读：{a.name}（2026-11-24）", "",
         f"- 候选逐图预测：`{os.path.relpath(a.probs, ROOT)}`",
         f"- valid：{len(idx):,} 个检查；聚合口径：侧位 0.6 → 检查内加权均值 → 患者收缩 0.85", "",
         "## 1　单模强度（E1）", "",
         f"- **宏 AUC = {cand_macro:.6f}**（门槛 ≥ {GATE_MACRO:.4f}）→ "
         f"{'✅ 过' if g1 else '❌ 未过'}",
         "- 逐类：" + " ".join(f"{v:.3f}" for v in per), "",
         "## 2　与三个 Tiny 的多样性（E2）", "",
         "| 对比对象 | Spearman |", "|---|---:|"]
    for n, s in spears.items():
        L.append(f"| `{n}` | {s:.3f} |")
    L += [f"| **均值** | **{mean_spear:.3f}**（门槛 < {GATE_SPEAR:.2f}）→ "
          f"{'✅ 够多样' if g2 else '❌ 太像'} |", "",
          "## 3　集成增益（E2）", "",
          f"- 现集成基线（3 Tiny 各 0.2833 + 供体 0.15）宏 AUC = **{base:.6f}**", "",
          "| 形态 | 权重/去掉 | Δ 宏 AUC |", "|---|---|---:|"]
    for w, d in add:
        L.append(f"| 添加 | {w:.2f} | {d:+.6f} |")
    for drop, d in repl:
        L.append(f"| 替换 | 去掉 `{drop}` | {d:+.6f} |")
    L += ["", f"- 最优添加：权重 {best_add[0]:.2f}，Δ = **{best_add[1]:+.6f}**"
              f"（门槛 ≥ +{GATE_DELTA:.4f}）→ {'✅ 过' if g3 else '❌ 未过'}",
          f"- 最优替换：去掉 `{best_repl[0]}`，Δ = {best_repl[1]:+.6f}" if best_repl[0] else "",
          "", "## 4　结论", ""]
    if g1 and g2 and g3:
        L.append("**三项全过：进入五折 OOF（E3）。**")
    elif not g1 and not g3:
        L.append("**单模与集成都未过：该线终止。**")
    else:
        L.append("**部分通过，需人工判断。** 优先看集成 Δ（它才是真正决定上线的那一项）。")

    out = OUT_MD.format(name=a.name)
    open(out, "w", encoding="utf-8").write("\n".join(L))
    json.dump({"name": a.name, "macro": cand_macro, "per_class": per,
               "spearman": spears, "mean_spearman": mean_spear, "base": base,
               "add": add, "replace": repl,
               "gates": {"macro": g1, "diversity": g2, "delta": g3}},
              open(out.replace(".md", ".json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n".join(L))
    print(f"\n  [wrote] {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
