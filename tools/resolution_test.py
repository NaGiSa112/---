"""Is input resolution the bottleneck for the weak classes?

The 512 cache throws away pixel detail: the originals are ~2500 px. Feeding 768 from the 512
cache would only interpolate, so every size here is decoded from the source image through
`load_square(path, size)` and normalised at that size — real detail, not upsampling.

Evaluated on fold 0 (a patient-disjoint holdout that oof_f0/last.pt never saw), per class, with
study-level AUC. No TTA and no donor, so the only variable is input size.

    python tools/resolution_test.py --sizes 512,544,640,768
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = r"E:\挑战赛"
sys.path.insert(0, os.path.join(ROOT, "src"))

from cxr.cache import image_path, load_square            # noqa: E402
from cxr.infer import load_checkpoint, preprocess       # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

NAMES = {1: "Atelectasis", 2: "Cardiomegaly", 3: "Effusion", 4: "Infiltration",
         5: "Mass/Nodule", 6: "Pneumonia", 7: "Pneumothorax", 8: "Consolidation",
         9: "Edema", 10: "No Finding"}


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="512,544,640,768")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "resolution_test.json"))
    a = ap.parse_args()
    sizes = [int(s) for s in a.sizes.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fold = pd.read_csv(os.path.join(ROOT, "04_新一轮", "00_规则", "patient_folds_5f.csv"))
    hold = set(fold[fold.fold == a.fold].Study_id.astype(str))
    tr = pd.read_csv(os.path.join(ROOT, "cxr_first", "train.csv"))
    Y = np.stack([parse_label(s) for s in tr.Predict_class])
    lab = pd.DataFrame(Y, columns=list(range(1, 11)))
    lab["Subject_id"] = tr.Subject_id.values
    lab["Study_id"] = tr.Study_id.values
    st_lab = lab.groupby(["Subject_id", "Study_id"])[list(range(1, 11))].max()
    st_lab = st_lab[st_lab.index.get_level_values(1).astype(str).isin(hold)]
    sub = tr[tr.Study_id.astype(str).isin(hold)].reset_index(drop=True)
    print(f"  fold {a.fold}: {len(sub)} 张图 / {len(st_lab)} 个 study")

    model, _ = load_checkpoint(os.path.join(ROOT, "runs", f"oof_f{a.fold}", "last.pt"), device)
    model.eval()

    res = {}
    for size in sizes:
        acc = {}
        for i, r in sub.iterrows():
            path = image_path("train", r.Subject_id, r.Study_id, r.Dicom_id)
            raw = load_square(path, size)          # 从原图解码到 size，不是缓存插值
            x = preprocess(raw, size, "per_image").unsqueeze(0).to(device)
            with torch.no_grad():
                p = torch.sigmoid(model(x))[0].cpu().numpy()
            key = (r.Subject_id, r.Study_id)
            acc[key] = acc.get(key, 0.0) + p
            if (i + 1) % 1500 == 0:
                print(f"    size {size}: {i+1}/{len(sub)}", flush=True)
        idx = st_lab.index.intersection(list(acc.keys()))
        # study 内平均
        per_study = {}
        for key in idx:
            per_study[key] = acc[key]
        cnt = sub.groupby(["Subject_id", "Study_id"]).size()
        rows = np.stack([per_study[k] / float(cnt.loc[k]) for k in idx])
        y = st_lab.loc[idx]
        per = [auc(y[c].to_numpy(), rows[:, c - 1]) for c in range(1, 11)]
        res[size] = {"per_class": per, "macro": float(np.nanmean(per))}
        print(f"  size {size:4d}: 宏平均 {np.nanmean(per):.4f}  " +
              " ".join(f"{v:.3f}" for v in per), flush=True)

    print()
    base = res[sizes[0]]["macro"]
    print("  %-18s %s %10s" % ("类别", " ".join(f"{s:>8}" for s in sizes), "Δ(最大)"))
    for c in range(1, 11):
        vals = [res[s]["per_class"][c - 1] for s in sizes]
        print("  %-18s %s %+10.4f" % (f"{c} {NAMES[c]}",
                                      " ".join(f"{v:8.4f}" for v in vals),
                                      vals[-1] - vals[0]))
    print("  %-18s %s %+10.4f" % ("宏平均", " ".join(f"{res[s]['macro']:8.4f}" for s in sizes),
                                  res[sizes[-1]]["macro"] - base))
    json.dump({str(k): v for k, v in res.items()}, open(a.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print()
    print(f"  [wrote] {a.out}")


if __name__ == "__main__":
    main()
