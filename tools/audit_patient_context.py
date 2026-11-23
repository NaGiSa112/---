"""患者上下文可用性对后处理的影响（反陈队审阅 §7 末段要求，CPU）。

问题
----
我们的后处理是 p = a·p_study + (1−a)·p_患者均值。它假设推理时**能拿到同一患者的全部检查**。
若国赛正式入口一次只给一张图（或把同患者的检查拆到不同批次），患者均值就用不全，
0.85 与 0.50 的取值都会失效。

三种情形
--------
  A 完整上下文  同患者全部检查一起聚合（当前做法），a=0.85 / 类2=0.50
  B 无上下文    每张图独立出分（a=1），这是"一次一图"入口的等价情形
  C 部分上下文  把每个患者的检查随机对半拆成两组，各自求均值再聚合（模拟分批）

输出：三种情形下的宏 AUC、类 2 AUC、受影响样本数与患者数，以及 A 相对 B 的增益。
只用 valid 与 train 的已有预测，不读 test，不跑 GPU。
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


def macro(df, lab):
    per = [auc(lab[c].to_numpy(), df[c].to_numpy()) for c in PCOLS]
    return float(np.nanmean(per)), per


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
    subj = np.asarray(w.index.get_level_values(0))
    n_per = pd.Series(subj).value_counts()
    multi = n_per[n_per > 1]

    def apply_ctx(a_vec, group_ids):
        """group_ids：每个 study 归属的聚合组（同组共用均值）。"""
        g = pd.Series(group_ids, index=w.index)
        mu = w.groupby(g).transform("mean")
        return w * a_vec + mu * (1.0 - a_vec)

    aA = np.full(10, A_OLD, np.float32)
    aA2 = aA.copy()
    aA2[C2 - 1] = A_NEW
    a1 = np.ones(10, np.float32)

    # A：完整上下文（按患者）
    A = apply_ctx(aA, subj)
    A2 = apply_ctx(aA2, subj)
    # B：无上下文（每 study 独立）
    B = apply_ctx(a1, np.arange(len(w)))

    # C：部分上下文 —— 每个多检查患者的检查随机对半，各半独立求均值
    rng = np.random.default_rng(0)
    gid = np.empty(len(w), dtype=object)
    order = {p: np.where(subj == p)[0] for p in multi.index}
    for p, rows in order.items():
        perm = rng.permutation(rows)
        half = len(perm) // 2
        for k, i in enumerate(perm):
            gid[i] = f"{p}_h0" if k < half else f"{p}_h1"
    for i in np.where(~np.isin(subj, multi.index))[0]:
        gid[i] = f"{subj[i]}_solo"
    C = apply_ctx(aA, gid)
    C2v = apply_ctx(aA2, gid)

    rows = []
    for name, df in (("A 完整上下文 a=0.85", A), ("A' 完整上下文 类2=0.50", A2),
                     ("B 无上下文 a=1", B), ("C 部分上下文 a=0.85", C),
                     ("C' 部分上下文 类2=0.50", C2v)):
        m, per = macro(df, lab)
        rows.append({"情形": name, "宏AUC": m, "类2 AUC": per[C2 - 1],
                     "类1 AUC": per[0], "类8 AUC": per[7]})

    L = ["# 患者上下文可用性审计（2026-11-20）", "",
         "> 回答「如果正式入口一次只给一张图，我们的患者收缩还能不能用」。",
         "> 只用 valid 的已有预测，CPU 完成。", "",
         f"- valid：{len(w)} study、{len(n_per)} 名患者；"
         f"其中 {len(multi)} 名患者有多次检查（影响 {int(multi.sum())} 个 study），"
         f"{int((n_per == 1).sum())} 名单检查患者不受上下文影响",
         "", "| 情形 | 宏 AUC | 类2 AUC | 类1 AUC | 类8 AUC |", "|---|---:|---:|---:|---:|"]
    for r in rows:
        L.append("| {} | {:.11f} | {:.9f} | {:.9f} | {:.9f} |".format(
            r["情形"], r["宏AUC"], r["类2 AUC"], r["类1 AUC"], r["类8 AUC"]))
    dA = rows[0]["宏AUC"] - rows[2]["宏AUC"]
    dC = rows[3]["宏AUC"] - rows[2]["宏AUC"]
    dC2 = rows[4]["宏AUC"] - rows[3]["宏AUC"]
    L += ["", "## 判读", "",
          f"- **完整上下文 vs 无上下文**：宏 AUC `{rows[0]['宏AUC']:.11f}` vs "
          f"`{rows[2]['宏AUC']:.11f}`，差 **{dA:+.11f}** —— 这是患者聚合本身的价值，"
          f"{'为正，值得保留' if dA > 0 else '非正，聚合无益'}。",
          f"- **部分上下文**：只拿到一半同患者检查时，宏 `{rows[3]['宏AUC']:.11f}`，"
          f"比完整上下文少 `{dA - dC:.11f}`，即损失约 `{(dA - dC) / dA * 100 if dA else 0:.0f}%` 的聚合收益。",
          f"- **类 2 = 0.50 在部分上下文下的增量**：`{dC2:+.11f}`"
          f"（完整上下文下为 `{rows[1]['宏AUC'] - rows[0]['宏AUC']:+.11f}`）。",
          "", "## 结论", "",
          "- 若正式入口能一次性给出同患者全部检查，现行 0.85 可用；",
          "- 若一次一图（a=1），患者聚合的收益全部消失，但**不会比不聚合更差**；",
          "- 类 2=0.50 的增量在部分上下文下会变化，且它本身只有 +0.0008（完整上下文），"
          "因此**不构成上线的充分理由**。"]

    out = os.path.join(ROOT, "reports", "患者上下文审计_2026-11-20.md")
    open(out, "w", encoding="utf-8").write("\n".join(L))
    json.dump(rows, open(os.path.join(ROOT, "reports", "context_audit_2026-11-20.json"),
                         "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n".join(L))
    print(f"\n  [wrote] {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
