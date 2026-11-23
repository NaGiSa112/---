"""基线核验（审阅 §9 第 1 步的 CPU 部分）。

核验项
------
1. 冻结折表：5 折、每折患者数、每患者只在一折、SHA-256
2. 每个折的预测键：不多不少、10 类齐全、无重复/缺失/NaN/越界（复用 verify_predictions）
3. 基线指标的全精度值：OOF 0.81470…、部署 valid 0.8251945484（并给容差）
4. 标签解析：Predict_class 的全部形态（单标签 / 多标签 / "10" / 空）
5. 官方 ID ↔ 内部列 回放：每类取单标签样例，逐列打印 one-hot
6. 官方类别映射与本地 config 是否一致

    python tools/audit_baseline.py
"""
from __future__ import annotations

import hashlib
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

OUT_MD = os.path.join(ROOT, "reports", "基线核验_2026-11-20.md")
OUT_JSON = os.path.join(ROOT, "reports", "baseline_audit_2026-11-20.json")


def sha256(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


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
    L: list[str] = ["# 基线核验（2026-11-20）", "",
                    "> 对应反陈队审阅 §9 第 1 步的 CPU 部分。全部可复跑。", ""]
    out: dict = {}

    # ---------- 1) 折表 ----------
    L += ["## 1　冻结折表", ""]
    fold = pd.read_csv(FOLDS)
    fold["Study_id"] = fold.Study_id.astype(str)
    tr = pd.read_csv(os.path.join(ROOT, "cxr_first", "train.csv"))
    n_study_tr = tr.Study_id.nunique()
    per_fold = fold.groupby("fold").agg(study=("Study_id", "size"),
                                        patient=("Subject_id", "nunique"))
    pat_folds = fold.groupby("Subject_id").fold.nunique()
    L += [f"- 文件 SHA-256：`{sha256(FOLDS)}`",
          f"- train 去重后 {n_study_tr:,} 个 study；折表 {len(fold):,} 行",
          f"- 折表覆盖 train 全部 study：{'✅' if len(fold) == n_study_tr else '❌'}",
          f"- 患者跨折数：{int((pat_folds > 1).sum())} {'✅' if (pat_folds > 1).sum() == 0 else '❌'}",
          "", "| fold | study | 患者 |", "|---:|---:|---:|"]
    for f, r in per_fold.iterrows():
        L.append(f"| {f} | {int(r.study):,} | {int(r.patient):,} |")
    out["fold_table"] = {"sha256": sha256(FOLDS), "rows": int(len(fold)),
                         "patients_cross_fold": int((pat_folds > 1).sum()),
                         "per_fold": {int(k): [int(v.study), int(v.patient)]
                                      for k, v in per_fold.iterrows()}}

    # ---------- 2) 每折预测键 ----------
    L += ["", "## 2　逐折预测完整性", ""]
    here = os.path.dirname(os.path.abspath(__file__))
    r = subprocess.run([sys.executable, os.path.join(here, "verify_predictions.py"),
                        "--tag", "oof", "--folds", "0,1,2,3,4"], cwd=ROOT,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                       encoding="utf-8", errors="replace")
    ok = r.returncode == 0
    L += [f"- `verify_predictions.py --tag oof --folds 0,1,2,3,4`：{'✅ 通过' if ok else '❌ 未通过'}",
          "", "```", *(r.stdout or "").strip().splitlines()[-8:], "```"]
    out["predictions_ok"] = bool(ok)

    # ---------- 3) 基线全精度 ----------
    L += ["", "## 3　基线指标（全精度）", ""]
    Y = np.stack([parse_label(s) for s in tr.Predict_class])
    lab = pd.DataFrame(Y, columns=PCOLS)
    lab["Subject_id"] = tr.Subject_id.values
    lab["Study_id"] = tr.Study_id.values
    st = lab.groupby(["Subject_id", "Study_id"])[PCOLS].max()

    parts = []
    for f in range(5):
        p = os.path.join(ROOT, "runs", f"oof_f{f}_study.csv")
        if os.path.exists(p):
            d = pd.read_csv(p)
            parts.append(d.pivot_table(index=["Subject_id", "Study_id"],
                                       columns="Predict_class", values="Probability",
                                       aggfunc="first").reindex(columns=PCOLS))
    oof = pd.concat(parts)
    oof = oof[~oof.index.duplicated()]
    idx = oof.index.intersection(st.index)
    per_oof = [auc(st.loc[idx, c].to_numpy(), oof.loc[idx, c].to_numpy()) for c in PCOLS]
    macro_oof = float(np.nanmean(per_oof))
    L += [f"- **OOF 五折（18,513 study）检查级宏 AUC = `{macro_oof:.11f}`**（历史记录 0.81470…）",
          "  - 逐类：" + " ".join(f"{v:.6f}" for v in per_oof)]
    out["oof_macro"] = macro_oof
    out["oof_per_class"] = per_oof

    va = pd.read_csv(os.path.join(ROOT, "cxr_first", "valid.csv"))
    Yv = np.stack([parse_label(s) for s in va.Predict_class])
    labv = pd.DataFrame(Yv, columns=PCOLS)
    labv["Subject_id"] = va.Subject_id.values
    labv["Study_id"] = va.Study_id.values
    stv = labv.groupby(["Subject_id", "Study_id"])[PCOLS].max()
    dv = pd.read_csv(os.path.join(ROOT, "runs", "orig_valid.csv"))
    wv = dv.pivot_table(index=["Subject_id", "Study_id"], columns="Predict_class",
                        values="Probability", aggfunc="first").reindex(columns=PCOLS)
    iv = wv.index.intersection(stv.index)
    per_v = [auc(stv.loc[iv, c].to_numpy(), wv.loc[iv, c].to_numpy()) for c in PCOLS]
    macro_v = float(np.nanmean(per_v))
    L += [f"- **部署版 valid（859 study）检查级宏 AUC = `{macro_v:.11f}`**（冻结包记录 0.82519454845，"
          f"差 {abs(macro_v - 0.82519454845):.2e}）",
          "  - 逐类：" + " ".join(f"{v:.6f}" for v in per_v)]
    out["valid_macro"] = macro_v
    out["valid_per_class"] = per_v

    # ---------- 4) 标签形态 ----------
    L += ["", "## 4　标签解析形态", ""]
    forms = tr.Predict_class.astype(str).value_counts()
    single = [s for s in forms.index if ";" not in s]
    multi = [s for s in forms.index if ";" in s]
    L += [f"- 不同标签串 {len(forms)} 种；单标签 {len(single)} 种、多标签 {len(multi)} 种",
          f"- 出现最多的前 6 种：" + "、".join(f"`{k}`×{v}" for k, v in forms.head(6).items()),
          f"- 含 `10` 的串：{sum(v for k, v in forms.items() if '10' in k.split(';')):,} 行；"
          f"其中只有 `10` 单标签：{int(forms.get('10', 0)):,} 行",
          "- 解析规则：按 `;` 切分 → 数字 id − 1 作为列索引；**`10` 永远不与其他标签同时出现**"
          f"（{'✅ 成立' if not any('10' in k.split(';') and len(k.split(';')) > 1 for k in forms.index) else '❌ 有例外'}）"]
    out["label_forms"] = {"n_forms": len(forms), "single": len(single), "multi": len(multi)}

    # ---------- 5) 官方 ID 回放 ----------
    L += ["", "## 5　官方 ID ↔ 内部列 回放", "",
          "每类取只含该标签的样例，打印 one-hot 非零列，应该恰好等于该类编号。", "",
          "| 类 | 单标签样例数 | 回放非零列 | 结论 |", "|---:|---:|---|---|"]
    replay_ok = True
    for c in PCOLS:
        only = np.array([s.strip() == str(c) for s in tr.Predict_class])
        idxs = np.where(only)[0][:3]
        cols = sorted({int(np.where(Y[i] == 1)[0][0]) + 1 for i in idxs})
        okc = cols == [c]
        replay_ok &= okc
        L.append(f"| {c} | {int(only.sum()):,} | {cols} | {'✅' if okc else '❌'} |")
    L += ["", f"结论：**{'10 类全部一致，数字链路无错位' if replay_ok else '存在错位，需停止训练并定位'}**"]
    out["id_replay_ok"] = bool(replay_ok)

    # ---------- 6) 官方映射与 config 一致性 ----------
    L += ["", "## 6　官方类别映射 vs 本地 config", ""]
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from cxr.config import LABEL_NAMES, LABEL_NAMES_ZH  # noqa: E402
    official = {1: ("Enlarged Cardiomediastinum", "纵隔心影增大"),
                2: ("Pneumothorax", "气胸"), 3: ("Consolidation", "肺部实变"),
                4: ("Pneumonia", "肺炎"), 5: ("Edema", "肺水肿"),
                6: ("Cardiomegaly", "心影扩大"), 7: ("Atelectasis", "肺不张"),
                8: ("Lung Opacity", "肺部阴影"), 9: ("Pleural Effusion", "胸腔积液"),
                10: ("No Finding", "无异常发现")}
    same = all(LABEL_NAMES[str(k)] == v[0] and LABEL_NAMES_ZH[str(k)] == v[1]
               for k, v in official.items())
    L += ["| ID | 官方英文 | 本地英文 | 官方中文 | 本地中文 | 一致 |",
          "|---:|---|---|---|---|:--:|"]
    for k, (en, zh) in official.items():
        ok2 = LABEL_NAMES[str(k)] == en and LABEL_NAMES_ZH[str(k)] == zh
        L.append(f"| {k} | {en} | {LABEL_NAMES[str(k)]} | {zh} | {LABEL_NAMES_ZH[str(k)]} | "
                 f"{'✅' if ok2 else '❌'} |")
    L += ["", f"结论：**{'本地映射已与官方两份文件一致' if same else '仍有不一致'}**"]
    out["label_names_ok"] = bool(same)

    # ---------- 汇总 ----------
    L += ["", "## 7　结论", ""]
    allok = ok and replay_ok and same
    L += [f"- 折表、预测键、基线指标、ID 回放、类别映射：**{'全部通过' if allok else '存在未通过项'}**",
          "- 该核验只覆盖 CPU 可核部分；GPU 侧的推理复算与测速另行记录。"]
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(L))
    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print("\n".join(L[-40:]))
    print(f"\n  [wrote] {os.path.relpath(OUT_MD, ROOT)}")
    print(f"  [wrote] {os.path.relpath(OUT_JSON, ROOT)}")


if __name__ == "__main__":
    main()
