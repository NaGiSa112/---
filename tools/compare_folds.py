"""任意候选 vs OOF 基线，同折配对比较。

    python tools/compare_folds.py --tag blend48

对照组 = runs/oof_f*（同日程、同折表、同 seed、no-KD、固定 last12）。
候选组 = runs/<tag>_f*（唯一差异是 --extra-channel asym）。

输出：
  逐折逐类 AUC
  合并 OOF 逐类 AUC 与 Δ
  患者级配对 bootstrap 的 Δ 95% 区间（配对 = 同一批患者重采样，两组用同一索引）
"""
from __future__ import annotations

import os
import sys

import argparse
import numpy as np
import pandas as pd

ROOT = r"E:\挑战赛"
PCOLS = list(range(1, 11))
NAMES = {1: "Atelectasis", 2: "Cardiomegaly", 3: "Effusion", 4: "Infiltration",
         5: "Mass/Nodule", 6: "Pneumonia", 7: "Pneumothorax", 8: "Consolidation",
         9: "Edema", 10: "No Finding"}

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


def wide(path: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    return d.pivot_table(index=["Subject_id", "Study_id"], columns="Predict_class",
                         values="Probability", aggfunc="first").reindex(columns=PCOLS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="asym")
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--folds", default="", help="留空=自动取已完成折；显式指定时报告会标注为部分折")
    ap.add_argument("--skip-integrity", action="store_true",
                    help="跳过完整性检查（不推荐，仅供事后复算旧产物）")
    _a = ap.parse_args()
    TAG = _a.tag
    if not _a.skip_integrity:
        # 审阅 §3.1：预测完整性不通过就拒绝比较，避免取交集掩盖漏预测
        folds = _a.folds or ",".join(
            str(f) for f in range(5)
            if os.path.exists(os.path.join(ROOT, "runs", f"{TAG}_f{f}_study.csv")))
        here = os.path.dirname(os.path.abspath(__file__))
        import subprocess
        rc = subprocess.run([sys.executable, os.path.join(here, "verify_predictions.py"),
                             "--tag", TAG, "--folds", folds], cwd=ROOT).returncode
        if rc != 0:
            raise SystemExit("[abort] 预测完整性检查未通过，拒绝比较")
    tr = pd.read_csv(os.path.join(ROOT, "cxr_first", "train.csv"))
    Y = np.stack([parse_label(s) for s in tr.Predict_class])
    lab = pd.DataFrame(Y, columns=PCOLS)
    lab["Subject_id"] = tr.Subject_id.values
    lab["Study_id"] = tr.Study_id.values
    st = lab.groupby(["Subject_id", "Study_id"])[PCOLS].max()
    fold = pd.read_csv(os.path.join(ROOT, "04_新一轮", "00_规则", "patient_folds_5f.csv"))
    fmap = dict(zip(fold.Study_id.astype(str), fold.fold.astype(int)))

    avail = [f for f in range(5)
             if os.path.exists(os.path.join(ROOT, "runs", f"{TAG}_f{f}_study.csv"))
             and os.path.exists(os.path.join(ROOT, "runs", f"oof_f{f}_study.csv"))]
    print(f"  可用折: {avail}")
    base = pd.concat([wide(os.path.join(ROOT, "runs", f"oof_f{f}_study.csv")) for f in avail])
    cand = pd.concat([wide(os.path.join(ROOT, "runs", f"{TAG}_f{f}_study.csv")) for f in avail])
    base = base[~base.index.duplicated()]
    cand = cand[~cand.index.duplicated()]
    idx = base.index.intersection(cand.index).intersection(st.index)
    base, cand, st = base.loc[idx], cand.loc[idx], st.loc[idx]
    print(f"  合并 {len(idx)} 个 study（{len(avail)} 折）")
    if len(avail) < 5:
        print(f"  ⚠️ 只用了 {len(avail)} 折，报告中只能称『部分折结果』，不能称完整 OOF")

    print()
    print("  %-18s %8s %8s %9s %10s" % ("类别", "基线", f"{TAG}", "Δ", "Δ范围(折间)"))
    per = {}
    for c in PCOLS:
        y = st[c].to_numpy()
        a = auc(y, base[c].to_numpy())
        b = auc(y, cand[c].to_numpy())
        per[c] = (a, b)
        # 逐折 Δ 的极值
        fd = []
        for f in avail:
            ids = set(fold[fold.fold == f].Study_id.astype(str))
            m = np.array([str(i[1]) in ids for i in idx])
            if m.sum() > 50:
                fd.append(auc(y[m], cand[c].to_numpy()[m]) - auc(y[m], base[c].to_numpy()[m]))
        rng = f"{min(fd):+.4f}~{max(fd):+.4f}" if fd else "—"
        print("  %-18s %8.4f %8.4f %+9.4f %10s" % (f"{c} {NAMES[c]}", a, b, b - a, rng))
    mA = float(np.nanmean([per[c][0] for c in PCOLS]))
    mB = float(np.nanmean([per[c][1] for c in PCOLS]))
    print("  %-18s %8.4f %8.4f %+9.4f" % ("宏平均", mA, mB, mB - mA))

    # ---- 患者级配对 bootstrap ----
    subj = np.array([i[0] for i in idx])
    uniq = np.unique(subj)
    rows_of = {u: np.where(subj == u)[0] for u in uniq}
    rng = np.random.default_rng(0)
    d_cls1, d_macro = [], []
    for _ in range(_a.boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        take = np.concatenate([rows_of[u] for u in pick])
        vals_a, vals_b = [], []
        for c in PCOLS:
            y = st[c].to_numpy()[take]
            if y.sum() == 0 or y.sum() == len(y):
                continue
            va = auc(y, base[c].to_numpy()[take])
            vb = auc(y, cand[c].to_numpy()[take])
            vals_a.append(va)
            vals_b.append(vb)
            if c == 1:
                d_cls1.append(vb - va)
        d_macro.append(float(np.nanmean(vals_b)) - float(np.nanmean(vals_a)))
    lo1, hi1 = np.percentile(d_cls1, [2.5, 97.5])
    loM, hiM = np.percentile(d_macro, [2.5, 97.5])
    print()
    print(f"  患者级配对 bootstrap（实际 {_a.boot} 次）")
    print("    类 1 Δ = %+.4f  95%% 区间 [%+.4f, %+.4f]  %s" % (
        per[1][1] - per[1][0], lo1, hi1,
        "✅ 下界 > 0" if lo1 > 0 else "❌ 区间含 0"))
    print("    宏平均 Δ = %+.4f  95%% 区间 [%+.4f, %+.4f]  %s" % (
        mB - mA, loM, hiM,
        "✅ 下界 > 0" if loM > 0 else "❌ 区间含 0"))


if __name__ == "__main__":
    main()
