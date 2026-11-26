"""E1/E2 判读 v2 —— 修正三处与部署不一致的地方。

修正内容（对应评审 2026-11-24 的四点）
--------------------------------------
1. **Spearman 用全部 10 类**：旧版只取了类 1 的排序相关（`cand[1]`），严重偏乐观。
   新版逐类算 Spearman 再取平均，同时给出"展平所有类"的参考值。
2. **融合顺序与部署一致**：旧版先做检查聚合+患者收缩，再 half_power 融合；
   实际部署是**逐图融合 → 检查聚合 → 患者收缩**。两者不可交换。
3. **替换测试用原权重**：旧版把候选设为 1.0 后再归一（实际给了 50%）；
   新版候选精确占据被移除 Tiny 的 0.283333 权重，其余成员权重不变。
4. 额外：先验证"成员逐图融合"能复现冻结包基线 0.82519454845，否则结果只作配对比较。

    python tools/e1_score_v2.py --probs runs/caformer_f2/valid_probs.csv --name caformer
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.environ.get("CXR_ROOT_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
try:
    from cxr.config import DATA as _DATA
except Exception:                                       # noqa: BLE001
    _DATA = os.path.join(ROOT, "cxr_first")

MEMBER_DIR = os.environ.get("CXR_MEMBER_DIR") or os.path.join(ROOT, "runs", "member_valid")
LABEL_CSV = os.path.join(_DATA, "valid.csv")
PCOLS = list(range(1, 11))
TINY = ["cnxt512_pat_e12", "cnxt512_pat_s43_e12", "tiny_seed44_last12"]
DONOR = "donor_v21"
W_TINY = 0.2833333333333333
W_DONOR = 0.15
FROZEN_BASELINE = 0.82519454845
GATE_MACRO = 0.8200
GATE_SPEAR = 0.90
GATE_DELTA = 0.0015

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


def load_image_probs(path: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    cols = [f"p{k}" for k in range(1, 11)]
    for c in cols:
        if c not in d.columns:
            raise SystemExit(f"[error] {path} 缺少列 {c}")
    return d[["Dicom_id", "Subject_id", "Study_id"] + cols].sort_values("Dicom_id").reset_index(drop=True)


def fuse_image_level(per_member: dict[str, pd.DataFrame], weights: dict[str, float]) -> pd.DataFrame:
    """部署口径：先在**逐图**层面 half_power 融合，再交给 aggregate()。"""
    ref = next(iter(per_member.values()))
    cols = [f"p{k}" for k in range(1, 11)]
    acc = np.zeros((len(ref), 10), np.float64)
    acc_sq = np.zeros((len(ref), 10), np.float64)
    wsum = 0.0
    for name, w in weights.items():
        d = per_member[name]
        if not d["Dicom_id"].equals(ref["Dicom_id"]):
            d = d.set_index("Dicom_id").loc[ref["Dicom_id"]].reset_index()
        p = d[cols].to_numpy(np.float64)
        acc += w * p
        acc_sq += w * np.square(p)
        wsum += w
    fused = 0.5 * (acc / wsum) + 0.5 * np.sqrt(acc_sq / wsum)
    out = ref[["Dicom_id", "Subject_id", "Study_id"]].copy()
    for i, c in enumerate(cols):
        out[c] = fused[:, i]
    return out


def aggregate(img: pd.DataFrame, vp: pd.DataFrame, lat: float = 0.6,
              shrink: float = 0.85) -> pd.DataFrame:
    d = img.merge(vp, on="Dicom_id", how="left")
    cols = [f"p{k}" for k in range(1, 11)]
    w = np.where(np.isin(d["ViewPosition"].fillna("NA").values, ["LATERAL", "LL"]), lat, 1.0)
    for c in cols:
        d[c] = d[c] * w
    g = d.groupby(["Subject_id", "Study_id"], as_index=False)[cols].sum()
    ws = pd.Series(w).groupby([d["Subject_id"].values, d["Study_id"].values]).sum()
    g[cols] = g[cols].div(ws.values[:, None])
    pm = g.groupby("Subject_id")[cols].transform("mean").values
    g[cols] = shrink * g[cols].values + (1 - shrink) * pm
    g.columns = ["Subject_id", "Study_id"] + PCOLS
    return g.set_index(["Subject_id", "Study_id"])


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

    per = {}
    for n in TINY + [DONOR]:
        p = os.path.join(MEMBER_DIR, f"{n}.csv")
        if os.path.exists(p):
            per[n] = load_image_probs(p)
    if not per:
        raise SystemExit("[error] 找不到成员逐图预测，先跑 tools/member_strength.py --predict")
    cand = load_image_probs(a.probs)

    base_w = {n: W_TINY for n in TINY if n in per}
    if DONOR in per:
        base_w[DONOR] = W_DONOR

    # 成员自检：能否复现冻结基线
    base_study = aggregate(fuse_image_level(per, base_w), vp).loc[lab.index]
    def macro(df):
        return float(np.nanmean([auc(lab[c].to_numpy(), df[c].to_numpy()) for c in PCOLS]))
    base_macro = macro(base_study)
    print(f"  成员逐图融合基线宏 AUC = {base_macro:.11f}（冻结包记录 {FROZEN_BASELINE:.11f}，"
          f"差 {base_macro - FROZEN_BASELINE:+.2e}）")

    # 候选单模（自身也要走完整口径）
    cand_study = aggregate(cand, vp).loc[lab.index]
    cand_macro = macro(cand_study)
    per_cls = [auc(lab[c].to_numpy(), cand_study[c].to_numpy()) for c in PCOLS]

    # 多样性：全部 10 类
    col = lambda df: df[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]].to_numpy()  # noqa: E731
    spears, flat = {}, {}
    for n in TINY:
        if n not in per:
            continue
        m = aggregate(per[n], vp).loc[lab.index]
        cs = [float(pd.Series(cand_study[k]).corr(pd.Series(m[k]), method="spearman"))
              for k in PCOLS]
        spears[n] = float(np.mean(cs))
        flat[n] = float(pd.Series(col(cand_study).ravel()).corr(
            pd.Series(col(m).ravel()), method="spearman"))
    mean_spear = float(np.mean(list(spears.values()))) if spears else float("nan")

    # 添加：候选按 w 并入，其余按比例缩到 1-w
    add = []
    for w in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
        wts = {k: v * (1 - w) for k, v in base_w.items()}
        wts[a.name] = w
        pooled = dict(per)
        pooled[a.name] = cand
        m = macro(aggregate(fuse_image_level(pooled, wts), vp).loc[lab.index])
        add.append((w, m - base_macro))

    # 替换：候选精确占据被移除 Tiny 的原权重
    repl = []
    for drop in TINY:
        if drop not in per:
            continue
        wts = {k: v for k, v in base_w.items() if k != drop}
        wts[a.name] = W_TINY
        pooled = dict(per)
        pooled[a.name] = cand
        m = macro(aggregate(fuse_image_level(pooled, wts), vp).loc[lab.index])
        repl.append((drop, m - base_macro))

    best_add = max(add, key=lambda r: r[1])
    best_repl = max(repl, key=lambda r: r[1]) if repl else (None, -9)
    g1, g2, g3 = cand_macro >= GATE_MACRO, mean_spear < GATE_SPEAR, \
        max(best_add[1], best_repl[1]) >= GATE_DELTA

    L = [f"# E1/E2 判读 v2：{a.name}（2026-11-24 修订版）", "",
         "- 融合顺序与部署一致：**逐图 half_power 融合 → 检查加权均值 → 患者收缩 0.85**",
         f"- 成员基线复现：{base_macro:.11f} vs 冻结 {FROZEN_BASELINE:.11f}"
         f"（差 {base_macro - FROZEN_BASELINE:+.2e}）", "",
         "## 1　单模强度", "",
         f"- **宏 AUC = {cand_macro:.6f}**（门槛 ≥{GATE_MACRO}）→ {'✅' if g1 else '❌'}",
         "- 逐类：" + " ".join(f"{v:.3f}" for v in per_cls), "",
         "## 2　多样性（**全部 10 类**，非仅类 1）", "",
         "| 对比对象 | 逐类 Spearman 均值 | 展平参考 |", "|---|---:|---:|"]
    for n in spears:
        L.append(f"| `{n}` | {spears[n]:.4f} | {flat[n]:.4f} |")
    L += [f"| **均值** | **{mean_spear:.4f}**（门槛 <{GATE_SPEAR}）→ "
          f"{'✅' if g2 else '❌'} | |", "",
          "## 3　集成增益（部署口径）", "",
          f"- 基线宏 AUC = **{base_macro:.6f}**", "",
          "| 形态 | 参数 | Δ 宏 AUC |", "|---|---|---:|"]
    for w, d in add:
        L.append(f"| 添加 | w={w:.2f} | {d:+.6f} |")
    for drop, d in repl:
        L.append(f"| 替换（原权重 {W_TINY:.4f}） | 去掉 `{drop}` | {d:+.6f} |")
    L += ["", f"- 最优添加：w={best_add[0]:.2f}，Δ = **{best_add[1]:+.6f}** "
              f"（门槛 ≥+{GATE_DELTA}）→ {'✅' if g3 else '❌'}",
          f"- 最优替换：去掉 `{best_repl[0]}`，Δ = {best_repl[1]:+.6f}" if best_repl[0] else "", "",
          "## 4　结论", ""]
    if g1 and g2 and g3:
        L.append("**三项全过，进入五折。**")
    else:
        fails = [n for n, ok in (("单模", g1), ("多样性", g2), ("集成增益", g3)) if not ok]
        L.append(f"**未通过：{'、'.join(fails)}。方向仍不支持采纳。**")

    out = os.path.join(ROOT, "reports", f"E1E2v2_判读_{a.name}.md")
    open(out, "w", encoding="utf-8").write("\n".join(L))
    json.dump({"name": a.name, "macro": cand_macro, "per_class": per_cls,
               "base_macro": base_macro, "frozen_baseline": FROZEN_BASELINE,
               "spearman_by_class": spears, "spearman_flat": flat,
               "mean_spearman": mean_spear, "add": add, "replace": repl,
               "gates": {"macro": g1, "diversity": g2, "delta": g3}},
              open(out.replace(".md", ".json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n".join(L))
    print(f"\n  [wrote] {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
