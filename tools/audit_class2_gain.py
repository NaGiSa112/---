"""类 2 收缩 0.50 的"救了谁"审计 + 与审阅给出的可验真数字对照。

审阅（反陈队 2026-11-20 §7）给出一个可精确验真的条件：若 valid 的类 2 有 37 个阳性检查、
822 个阴性（K = 37×822 = 30,414 对），且宏差显示为 0.00083，则"净纠正对数"B 必须满足
    round5(0.8251945484 + B/(10K)) = 0.82603  且  round5(B/(10K)) = 0.00083
枚举后只剩 253 或 253.5（无平局跃迁时只能是 253）。

本脚本从真实预测复算：翻转对计数、按患者归因、逐患者剔除后的 Δ、频次与视角交叉表。
只用 train/valid，不读 test。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = r"E:\挑战赛"
PCOLS = list(range(1, 11))
C2 = 2
A_OLD, A_NEW = 0.85, 0.50
NAMES = {1: "Enlarged Cardiomediastinum", 2: "Pneumothorax", 3: "Consolidation",
         4: "Pneumonia", 5: "Edema", 6: "Cardiomegaly", 7: "Atelectasis",
         8: "Lung Opacity", 9: "Pleural Effusion", 10: "No Finding"}

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


def main() -> None:
    va = pd.read_csv(os.path.join(ROOT, "cxr_first", "valid.csv"))
    Y = np.stack([parse_label(s) for s in va.Predict_class])
    lab = pd.DataFrame(Y, columns=PCOLS)
    lab["Subject_id"] = va.Subject_id.values
    lab["Study_id"] = va.Study_id.values
    lab = lab.groupby(["Subject_id", "Study_id"])[PCOLS].max()

    d = pd.read_csv(os.path.join(ROOT, "runs", "valid_noshrink.csv"))
    w = d.pivot_table(index=["Subject_id", "Study_id"], columns="Predict_class",
                      values="Probability", aggfunc="first").reindex(columns=PCOLS)
    idx = w.index.intersection(lab.index)
    w, lab = w.loc[idx], lab.loc[idx]

    subj = w.index.get_level_values(0)
    mu = w.groupby(subj).transform("mean")
    q_old = A_OLD * w + (1 - A_OLD) * mu
    q_new = A_NEW * w + (1 - A_NEW) * mu

    y = lab[C2].to_numpy()
    s_old = q_old[C2].to_numpy()
    s_new = q_new[C2].to_numpy()
    pos = y > 0
    print(f"  类 {C2} {NAMES[C2]}：{len(y)} 个 study，阳性 {int(pos.sum())}、"
          f"阴性 {int((~pos).sum())}，对 {int(pos.sum()) * int((~pos).sum())}")
    print(f"  AUC（0.85）= {auc(y, s_old):.10f}")
    print(f"  AUC（0.50）= {auc(y, s_new):.10f}")
    print(f"  Δ = {auc(y, s_new) - auc(y, s_old):+.10f}")

    # ---- 逐对翻转计数（平局计半）----
    sp, sn = s_new[pos], s_new[~pos]
    op, on = s_old[pos], s_old[~pos]
    # 用广播矩阵（37×822 很小）
    new_win = (sp[:, None] > sn[None, :]).sum() + 0.5 * (sp[:, None] == sn[None, :]).sum()
    old_win = (op[:, None] > on[None, :]).sum() + 0.5 * (op[:, None] == on[None, :]).sum()
    B = float(new_win - old_win)
    print(f"\n  实际净纠正对数 B = {B:+.1f}（满分对等价量）")
    print(f"  由 Δ 反推：B = Δ×对总数 = {(auc(y, s_new) - auc(y, s_old)) * pos.sum() * (~pos).sum():+.1f}")

    # ---- 与审阅的可验真条件对照 ----
    K = int(pos.sum()) * int((~pos).sum())
    macro_old_txt = 0.8251945484
    print(f"\n  === 与审阅 §7 的可验真条件对照（K = {K}）")
    print(f"  审阅要求：B ∈ {{253, 253.5}}（无平局跃迁时应为 253）")
    print(f"  实测：B = {B:.1f}  → {'✅ 吻合' if abs(B - 253) < 1 or abs(B - 253.5) < 1 else '❌ 不吻合'}")
    for Bc in (252.5, 253.0, 253.5, 254.0):
        print(f"    若 B={Bc:5.1f}：Δclass2={Bc/K:+.10f}  Δmacro={Bc/(10*K):+.10f}"
              f"  新宏={macro_old_txt + Bc/(10*K):.10f}")

    # ---- 谁贡献：按患者 ----
    pid = w.index.get_level_values(0)
    rows = []
    for i in np.where(pos)[0]:
        p_i = s_new[i] - s_old[i]
        d_i = (s_new[i] > sn).sum() + 0.5 * (s_new[i] == sn).sum() \
            - ((s_old[i] > on).sum() + 0.5 * (s_old[i] == on).sum())
        rows.append((pid[i], w.index[i][1], float(p_i), float(d_i)))
    pat = pd.DataFrame(rows, columns=["patient", "study", "dscore", "dpairs"])
    print("\n  === 按阳性检查的净贡献（前 8）")
    print("  %-14s %-14s %10s %10s" % ("patient", "study", "Δ分数", "净纠正对"))
    for _, r in pat.sort_values("dpairs", ascending=False).head(8).iterrows():
        print("  %-14s %-14s %+10.5f %+10.1f" % (str(r.patient)[:14], str(r.study)[:14],
                                                 r.dscore, r.dpairs))
    print("  负贡献的阳性个数:", int((pat.dpairs < 0).sum()),
          " 零贡献:", int((pat.dpairs == 0).sum()))
    by_pat = pat.groupby("patient").dpairs.sum().sort_values(ascending=False)
    print(f"  涉及阳性患者 {len(by_pat)} 名；净贡献 >0 的 {int((by_pat > 0).sum())} 名")

    # ---- 逐患者剔除 ----
    print("\n  === 逐一剔除净贡献最大的阳性患者后，类 2 Δ 与宏 Δ")
    base_macro = None
    for k, (p, contrib) in enumerate(by_pat.head(5).items(), 1):
        keep = pid != p
        if pos[keep].sum() == 0:
            continue
        dd = auc(y[keep], s_new[keep]) - auc(y[keep], s_old[keep])
        flag = "同向" if dd > 0 else "**反向**"
        print(f"  剔除 {str(p)[:14]}（贡献 {contrib:+.1f} 对）→ 类2 Δ = {dd:+.10f}  {flag}")

    # ---- 视角与频次交叉 ----
    views = va.groupby("Study_id").ViewPosition.first()
    pat["view"] = [views.get(s, "NA") for s in pat.study]
    print("\n  === 获益阳性的视角分布")
    print("  " + str(pat.groupby("view").dpairs.agg(["count", "sum"]).round(2).to_dict("index")))


if __name__ == "__main__":
    main()
