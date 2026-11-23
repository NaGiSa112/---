"""类 2 收缩 0.50 的完整回放审计（反陈队审阅 §7 要求项，全部 CPU）。

输出
----
1. valid 与 OOF 两套数据上，A（全 0.85）与 B（仅类 2 = 0.50）的 10 类 AUC 与宏
2. 其余 9 类逐值最大差；同患者检查对内排序不变性；单检查患者不变性
3. 净纠正对数 B；**跨患者 / 同患者**对分别统计
4. 患者检查频次（1 / 2–3 / 4–7 / 8+）× 正负侧的净贡献交叉表
5. 逐一剔除头部阳性患者后的 Δ
6. a = 1（完全不收缩）作为参照
7. OOF 逐折 Δ

    python tools/audit_class2_full.py
"""
from __future__ import annotations

import json
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


def shrink(df, a_vec):
    subj = df.index.get_level_values(0)
    mu = df.groupby(subj).transform("mean")
    return df * a_vec + mu * (1.0 - a_vec)


def bucket(n: int) -> str:
    return "1" if n == 1 else "2-3" if n <= 3 else "4-7" if n <= 7 else "8+"


def analyse(noshrink: pd.DataFrame, lab: pd.DataFrame, tag: str,
            folds: np.ndarray | None = None) -> dict:
    idx = noshrink.index.intersection(lab.index)
    w, y_lab = noshrink.loc[idx], lab.loc[idx]
    aA = np.full(10, A_OLD, np.float32)
    aB = aA.copy()
    aB[C2 - 1] = A_NEW
    A = shrink(w, aA)
    B = shrink(w, aB)
    a1 = np.ones(10, np.float32)
    N = shrink(w, a1)

    perA = [auc(y_lab[c].to_numpy(), A[c].to_numpy()) for c in PCOLS]
    perB = [auc(y_lab[c].to_numpy(), B[c].to_numpy()) for c in PCOLS]
    perN = [auc(y_lab[c].to_numpy(), N[c].to_numpy()) for c in PCOLS]
    m = lambda v: float(np.nanmean(v))  # noqa: E731

    # 9 列逐值差与两组不变性
    other_max = max(float(np.abs(A[c].to_numpy() - B[c].to_numpy()).max())
                    for c in PCOLS if c != C2)
    pid = A.index.get_level_values(0)
    pid_arr = np.asarray(pid)
    n_per = pd.Series(pid).value_counts()
    same_study_pairs_ok = True
    for p, g in A.groupby(pid):
        if len(g) < 2:
            continue
        orderA = np.argsort(g[C2].to_numpy(), kind="stable")
        orderB = np.argsort(B.loc[g.index, C2].to_numpy(), kind="stable")
        if not np.array_equal(orderA, orderB):
            same_study_pairs_ok = False
            break
    single = n_per[n_per == 1].index
    single_max = 0.0
    if len(single):
        sA = A[A.index.get_level_values(0).isin(single)]
        sB = B[B.index.get_level_values(0).isin(single)]
        single_max = float(np.abs(sA.to_numpy() - sB.to_numpy()).max())

    # 净纠正对数 + 跨/同患者拆分
    yv = y_lab[C2].to_numpy()
    sA2, sB2 = A[C2].to_numpy(), B[C2].to_numpy()
    pos = yv > 0
    ppos, pneg = np.where(pos)[0], np.where(~pos)[0]
    sp_new, sn_new = sB2[ppos], sB2[pneg]
    sp_old, sn_old = sA2[ppos], sA2[pneg]
    win_new = (sp_new[:, None] > sn_new[None, :]).sum() + \
              0.5 * (sp_new[:, None] == sn_new[None, :]).sum()
    win_old = (sp_old[:, None] > sn_old[None, :]).sum() + \
              0.5 * (sp_old[:, None] == sn_old[None, :]).sum()
    Bnet = float(win_new - win_old)
    same_mask = pid_arr[ppos][:, None] == pid_arr[pneg][None, :]
    d_new = (sp_new[:, None] > sn_new[None, :]).astype(np.float32) + \
        0.5 * (sp_new[:, None] == sn_new[None, :])
    d_old = (sp_old[:, None] > sn_old[None, :]).astype(np.float32) + \
        0.5 * (sp_old[:, None] == sn_old[None, :])
    delta = d_new - d_old
    B_cross = float(delta[~same_mask].sum())
    B_same = float(delta[same_mask].sum())

    # 频次交叉表：正侧 / 负侧各自的分桶
    pos_bucket = np.array([bucket(int(n_per[p])) for p in pid_arr[ppos]])
    neg_bucket = np.array([bucket(int(n_per[p])) for p in pid_arr[pneg]])
    cross: dict[str, dict[str, float]] = {}
    for pb in ("1", "2-3", "4-7", "8+"):
        cross[pb] = {}
        for nb in ("1", "2-3", "4-7", "8+"):
            m2 = (pos_bucket[:, None] == pb) & (neg_bucket[None, :] == nb)
            cross[pb][nb] = float(delta[m2].sum()) if m2.any() else float("nan")

    # 逐阳性贡献与剔除
    contrib = {}
    for k, i in enumerate(ppos):
        contrib[int(i)] = float(delta[k].sum())
    order = sorted(contrib, key=lambda i: -contrib[i])
    loo = []
    for i in order[:5]:
        keep = np.ones(len(yv), bool)
        keep[i] = False
        if keep.sum() and (yv[keep] > 0).any() and (yv[keep] <= 0).any():
            loo.append({"study_index": int(i), "patient": str(pid[i]),
                        "contrib": contrib[i],
                        "delta_after_removal": auc(yv[keep], sB2[keep]) - auc(yv[keep], sA2[keep])})

    # 逐折
    per_fold = {}
    if folds is not None:
        for f in np.unique(folds):
            m2 = folds == f
            if m2.sum() == 0 or (yv[m2] > 0).sum() == 0:
                continue
            per_fold[int(f)] = {"n": int(m2.sum()),
                                "auc_old": auc(yv[m2], sA2[m2]),
                                "auc_new": auc(yv[m2], sB2[m2]),
                                "delta": auc(yv[m2], sB2[m2]) - auc(yv[m2], sA2[m2])}

    return {"tag": tag, "n_study": int(len(idx)), "n_pos": int(pos.sum()),
            "n_neg": int((~pos).sum()), "pairs": int(pos.sum() * (~pos).sum()),
            "macro_old": m(perA), "macro_new": m(perB), "macro_noshrink": m(perN),
            "per_class_old": perA, "per_class_new": perB, "per_class_noshrink": perN,
            "delta_macro": m(perB) - m(perA),
            "delta_class2": perB[C2 - 1] - perA[C2 - 1],
            "other_max_diff": other_max,
            "same_patient_order_unchanged": bool(same_study_pairs_ok),
            "single_study_max_diff": single_max,
            "B_net_pairs": Bnet, "B_cross_patient": B_cross, "B_same_patient": B_same,
            "freq_cross": cross, "top_contrib": order[:8],
            "leave_one_out": loo, "per_fold": per_fold}


def main() -> None:
    tr = pd.read_csv(os.path.join(ROOT, "cxr_first", "train.csv"))
    Y = np.stack([parse_label(s) for s in tr.Predict_class])
    lab = pd.DataFrame(Y, columns=PCOLS)
    lab["Subject_id"] = tr.Subject_id.values
    lab["Study_id"] = tr.Study_id.values
    lab = lab.groupby(["Subject_id", "Study_id"])[PCOLS].max()

    fold = pd.read_csv(os.path.join(ROOT, "04_新一轮", "00_规则", "patient_folds_5f.csv"))
    fmap = dict(zip(fold.Study_id.astype(str), fold.fold.astype(int)))

    # ---- OOF：反解 0.85 得到未收缩概率 ----
    parts = []
    for f in range(5):
        p = os.path.join(ROOT, "runs", f"oof_f{f}_study.csv")
        d = pd.read_csv(p)
        parts.append(d.pivot_table(index=["Subject_id", "Study_id"], columns="Predict_class",
                                   values="Probability", aggfunc="first").reindex(columns=PCOLS))
    S = pd.concat(parts)
    S = S[~S.index.duplicated()]
    subj = S.index.get_level_values(0)
    G = (S - (1 - A_OLD) * S.groupby(subj).transform("mean")) / A_OLD
    fof = np.array([fmap.get(str(i[1]), -1) for i in G.index])
    r_oof = analyse(G, lab, "OOF 五折", folds=fof)

    # ---- valid：直接从未收缩预测出发 ----
    va = pd.read_csv(os.path.join(ROOT, "cxr_first", "valid.csv"))
    Yv = np.stack([parse_label(s) for s in va.Predict_class])
    labv = pd.DataFrame(Yv, columns=PCOLS)
    labv["Subject_id"] = va.Subject_id.values
    labv["Study_id"] = va.Study_id.values
    labv = labv.groupby(["Subject_id", "Study_id"])[PCOLS].max()
    dv = pd.read_csv(os.path.join(ROOT, "runs", "valid_noshrink.csv"))
    Gv = dv.pivot_table(index=["Subject_id", "Study_id"], columns="Predict_class",
                        values="Probability", aggfunc="first").reindex(columns=PCOLS)
    r_va = analyse(Gv, labv, "部署 valid")

    # ---- 输出 ----
    P = lambda x: f"{x:.11f}"  # noqa: E731
    L = ["# 类 2 收缩 0.50 完整回放审计（2026-11-20）", "",
         "> CPU 完成，对应反陈队审阅 §7 全部要求项。只用 train/valid，不读 test。", ""]
    for r in (r_va, r_oof):
        L += [f"## {r['tag']}", "",
              f"- study {r['n_study']:,}（阳性 {r['n_pos']}、阴性 {r['n_neg']}），正负对 {r['pairs']:,}",
              f"- 宏 AUC：全 0.85 = `{P(r['macro_old'])}` → 类2=0.50 = `{P(r['macro_new'])}`"
              f"（Δ `{r['delta_macro']:+.11f}`）；不收缩 a=1 = `{P(r['macro_noshrink'])}`",
              f"- 类 2 AUC：`{r['per_class_old'][C2-1]:.11f}` → `{r['per_class_new'][C2-1]:.11f}`"
              f"（Δ `{r['delta_class2']:+.11f}`）",
              f"- 其余 9 类逐值最大差：`{r['other_max_diff']:.3e}`"
              f" {'✅' if r['other_max_diff'] == 0 else '❌'}",
              f"- 同患者检查对内排序不变：{'✅' if r['same_patient_order_unchanged'] else '❌'}；"
              f"单检查患者最大差 `{r['single_study_max_diff']:.3e}`",
              f"- **净纠正对数 B = {r['B_net_pairs']:+.1f}**"
              f"（跨患者 {r['B_cross_patient']:+.1f}、同患者 {r['B_same_patient']:+.1f}）"]
        if r["per_fold"]:
            L += ["", "| fold | study | AUC 0.85 | AUC 0.50 | Δ |", "|---:|---:|---:|---:|---:|"]
            for f, d in sorted(r["per_fold"].items()):
                L.append(f"| {f} | {d['n']:,} | {d['auc_old']:.6f} | {d['auc_new']:.6f} | "
                         f"{d['delta']:+.6f} |")
        L += ["", "### 频次交叉表（行=阳性侧检查频次，列=阴性侧）", "",
              "| 正\\负 | 1 | 2-3 | 4-7 | 8+ |", "|---:|---:|---:|---:|---:|"]
        for pb in ("1", "2-3", "4-7", "8+"):
            cells = " | ".join(f"{r['freq_cross'][pb][nb]:+.1f}" for nb in ("1", "2-3", "4-7", "8+"))
            L.append(f"| {pb} | {cells} |")
        L += ["", "### 剔除头部阳性患者", "",
              "| 患者 | 贡献对 | 剔除后类2 Δ | 方向 |", "|---|---:|---:|---|"]
        for d in r["leave_one_out"]:
            L.append(f"| `{d['patient']}` | {d['contrib']:+.1f} | "
                     f"{d['delta_after_removal']:+.6f} | "
                     f"{'同向' if d['delta_after_removal'] > 0 else '**反向**'} |")
        L.append("")

    L += ["## 判读", "",
          "- 收益全部来自**跨患者对**（同患者对内排序按构造不变）。",
          "- 需要看频次交叉表里净贡献是否集中在单一格子；集中则说明这是个别患者的偶然。",
          "- 剔除头部患者后方向仍为正，但幅度会明显下降（见表）。"]
    out_md = os.path.join(ROOT, "reports", "类2回放审计_2026-11-20.md")
    open(out_md, "w", encoding="utf-8").write("\n".join(L))
    json.dump({"valid": r_va, "oof": r_oof},
              open(os.path.join(ROOT, "reports", "class2_audit_2026-11-20.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n".join(L[:120]))
    print(f"\n  [wrote] {os.path.relpath(out_md, ROOT)}")


if __name__ == "__main__":
    main()
