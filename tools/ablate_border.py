"""Ablation: does the model's border/text attention actually cost AUC?

Motivation
----------
The class-1 evidence figure shows blobs sitting on burned-in annotations (`SUPINE`, `PORTABLE`,
`L`, circular markers) inside the outer 8% band. Quantified over 120 positives per class, the
band carries 3-16% of each class's evidence. If that attention is a shortcut, removing the band
should improve the classes that lean on it.

Two variants on the official valid split, same weights, same geometry:
    base   原图
    mask8  宽 8% 的外框像素置零（归一化前）
    crop84 取中心 84% 再放大回 544（等价于去掉边框同时保留视野）

    python tools/ablate_border.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image

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
IMG = 544
CACHE = 512


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


def spatial_logits(model, x):
    feat = model._spatial_features(x)
    w, b = model.head[1].weight, model.head[1].bias
    return torch.einsum("bchw,oc->bohw", feat.float(), w) + b.view(1, -1, 1, 1)


def pooled(model, x):
    """The model's own study logits (LSE head), not the spatial map."""
    return model(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy", default=os.path.join(ROOT, "model", "deploy_final_7851.json"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = json.load(open(a.deploy, encoding="utf-8"))
    members = [m for m in spec["members"] if m.get("kind") != "donor_v21"]
    print(f"  成员 {len(members)} 个（供体不参与本消融，保持变量单一）")

    va = pd.read_csv(os.path.join(ROOT, "cxr_first", "valid.csv"))
    if a.limit:
        va = va.head(a.limit)
    Y = np.stack([parse_label(s) for s in va.Predict_class])
    lab = pd.DataFrame(Y, columns=list(range(1, 11)))
    lab["Subject_id"] = va.Subject_id.values
    lab["Study_id"] = va.Study_id.values
    st_lab = lab.groupby(["Subject_id", "Study_id"])[list(range(1, 11))].max()

    models = []
    for m in members:
        mod, _ = load_checkpoint(os.path.join(ROOT, m["ckpt"].replace("/", os.sep)), device)
        mod.eval()
        models.append(mod)

    def forward_variant(raw: np.ndarray, mode: str) -> np.ndarray:
        if mode == "crop84":
            h = raw.shape[0]
            m = int(round(0.08 * h))
            raw = raw[m:h - m, m:h - m]
            raw = np.asarray(Image.fromarray(raw).resize((CACHE, CACHE), Image.BILINEAR), np.float32)
        elif mode == "mask8":
            raw = raw.copy()
            h, w = raw.shape
            mh, mw = int(round(0.08 * h)), int(round(0.08 * w))
            raw[:mh, :] = 0
            raw[-mh:, :] = 0
            raw[:, :mw] = 0
            raw[:, -mw:] = 0
        x = preprocess(raw, IMG, "per_image").unsqueeze(0).to(device)
        acc = []
        for mod in models:
            with torch.no_grad():
                acc.append(torch.sigmoid(mod(x))[0].cpu().numpy())
        return np.mean(acc, axis=0)

    recs = {mode: [] for mode in ("base", "mask8", "crop84")}
    for i, r in va.reset_index(drop=True).iterrows():
        path = image_path("valid", r.Subject_id, r.Study_id, r.Dicom_id)
        raw = load_square(path, CACHE)
        for mode in recs:
            p = forward_variant(raw, mode)
            recs[mode].append({"Subject_id": r.Subject_id, "Study_id": r.Study_id,
                               **{c: float(p[c - 1]) for c in range(1, 11)}})
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(va)}")

    out = {"per_class": {}, "macro": {}}
    wide = {}
    for mode, rows in recs.items():
        d = pd.DataFrame(rows).groupby(["Subject_id", "Study_id"]).mean()
        d = d.loc[st_lab.index.intersection(d.index)]
        wide[mode] = d
        per = [auc(st_lab[c].loc[d.index].to_numpy(), d[c].to_numpy()) for c in range(1, 11)]
        out["per_class"][mode] = per
        out["macro"][mode] = float(np.nanmean(per))

    print()
    print("  %-18s %8s %8s %8s %10s %10s" % ("类别", "base", "mask8", "crop84",
                                              "mask8−base", "crop−base"))
    for c in range(1, 11):
        b, m8, cr = (out["per_class"][k][c - 1] for k in ("base", "mask8", "crop84"))
        print("  %-18s %8.4f %8.4f %8.4f %+10.4f %+10.4f" % (
            f"{c} {NAMES[c]}", b, m8, cr, m8 - b, cr - b))
    print("  %-18s %8.4f %8.4f %8.4f %+10.4f %+10.4f" % (
        "宏平均", out["macro"]["base"], out["macro"]["mask8"], out["macro"]["crop84"],
        out["macro"]["mask8"] - out["macro"]["base"],
        out["macro"]["crop84"] - out["macro"]["base"]))

    path = os.path.join(ROOT, "reports", "outside_body", "border_ablation.json")
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print()
    print(f"  [wrote] {path}")


if __name__ == "__main__":
    main()
