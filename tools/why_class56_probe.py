"""Is the class-5/6 heatmap difference caused by the weights, or by the estimator?

The comparison figure averages only 8 positives per class. A mean heatmap over 8 images is
itself a noisy estimate: draw 8 different positives and the average changes shape.

This script measures the estimator's own noise and compares it with the cross-model
difference, using the same 40 positives / 40 negatives per class:

  r_self(orig)   two disjoint halves of the SAME model's positives
  r_self(new)    same, for the retrained model
  r_cross        all 40, original vs retrained

If r_cross is close to the r_self values, the visible difference is sampling noise in the
figure, not a different thing the models look at.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.join(r"E:\挑战赛", "src"))

from cxr.cache import image_path, load_square          # noqa: E402
from cxr.infer import load_checkpoint, preprocess     # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = r"E:\挑战赛"
NAMES = {1: "Atelectasis", 2: "Cardiomegaly", 3: "Effusion", 4: "Infiltration",
         5: "Mass/Nodule", 6: "Pneumonia", 7: "Pneumothorax", 8: "Consolidation",
         9: "Edema", 10: "No Finding"}
CLASSES = list(range(1, 11))
N_POS = 20
N_NEG = 20
IMG = 544
CACHE = 512


def parse_label(s) -> np.ndarray:
    y = np.zeros(10, np.float32)
    s = str(s).strip()
    if s == "" or s.lower() == "nan":
        return y
    cs = [c.strip() for c in s.split(";") if c.strip()]
    if cs == ["10"]:
        y[9] = 1.0
        return y
    for c in cs:
        i = int(c) - 1
        if i < 9:
            y[i] = 1.0
    return y


def spatial_logits(model, x):
    """Per-position class logits B x 10 x h x w — identical to tools/heatmap_probe.py."""
    feat = model._spatial_features(x)
    w, b = model.head[1].weight, model.head[1].bias
    return torch.einsum("bchw,oc->bohw", feat.float(), w) + b.view(1, -1, 1, 1)


def heat_of(model, path, cls, device):
    raw = load_square(path, CACHE)
    x = preprocess(raw, IMG, "per_image").unsqueeze(0).to(device)
    with torch.no_grad():
        m = spatial_logits(model, x)
    h = F.relu(m[0, cls - 1]).cpu().numpy()
    h = (h - h.min()) / (h.max() - h.min() + 1e-9)
    return np.asarray(Image.fromarray((h * 255).astype(np.uint8))
                      .resize((IMG, IMG), Image.BILINEAR), np.float32) / 255.0


def corr(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(os.path.join(ROOT, "cxr_first", "valid.csv"))
    Y = np.stack([parse_label(s) for s in df.Predict_class])
    rng = np.random.default_rng(7)

    models = {}
    for tag, ck in (("orig", "runs/cnxt512_pat_e12/best.pt"), ("new", "runs/repro42/best.pt")):
        models[tag], _ = load_checkpoint(os.path.join(ROOT, ck.replace("/", os.sep)), device)
        models[tag].eval()
        print(f"  已加载 {tag}: {ck}")

    rows = []
    for cls in CLASSES:
        pos = np.where(Y[:, cls - 1] > 0)[0]
        neg = np.where(Y[:, cls - 1] <= 0)[0]
        pi = rng.choice(pos, min(N_POS, len(pos)), replace=False)
        ni = rng.choice(neg, min(N_NEG, len(neg)), replace=False)

        maps = {}
        for tag in ("orig", "new"):
            acc_p, acc_n = [], []
            for i in pi:
                r = df.iloc[i]
                acc_p.append(heat_of(models[tag],
                                     image_path("valid", r.Subject_id, r.Study_id, r.Dicom_id),
                                     cls, device))
            for i in ni:
                r = df.iloc[i]
                acc_n.append(heat_of(models[tag],
                                     image_path("valid", r.Subject_id, r.Study_id, r.Dicom_id),
                                     cls, device))
            maps[tag] = (np.stack(acc_p), np.stack(acc_n))
        print(f"  类 {cls} {NAMES[cls]}：正样本 {len(pi)} / 负样本 {len(ni)} 张已算完")

        def mean_diff(tag, idx):
            p, n = maps[tag]
            return np.clip(p[idx].mean(0) - n.mean(0), 0, None)

        half = len(pi) // 2
        a_idx = np.arange(half)
        b_idx = np.arange(half, len(pi))
        r_self_orig = corr(mean_diff("orig", a_idx), mean_diff("orig", b_idx))
        r_self_new = corr(mean_diff("new", a_idx), mean_diff("new", b_idx))
        all_idx = np.arange(len(pi))
        r_cross = corr(mean_diff("orig", all_idx), mean_diff("new", all_idx))
        # 只用原图里 8 张时，两次独立的 8 张抽样能有多一致（复现图上的情形）
        r8_a = corr(mean_diff("orig", np.arange(8)), mean_diff("new", np.arange(8)))

        print(f"    同一模型、两半样本互比   orig={r_self_orig:.3f}  new={r_self_new:.3f}")
        print(f"    两个模型、同一批 {len(pi)} 张      r_cross={r_cross:.3f}")
        print(f"    两个模型、各取 8 张（图中设置） r_8={r8_a:.3f}")
        verdict = ("两模型一致" if r_cross > r_self_orig and r_cross > r_self_new
                   else "跨模型差异大于估计噪声" if r_cross < 0.85 else "跨模型略低于自比")
        print(f"    判定：{verdict}")
        print()
        rows.append({"class": cls, "name": NAMES[cls], "n_pos": len(pi), "n_neg": len(ni),
                     "r_self_orig": r_self_orig, "r_self_new": r_self_new,
                     "r_cross": r_cross, "r_cross_8": r8_a})

    import json
    out = os.path.join(ROOT, "reports", "heatmaps_retrain", "map_agreement.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[wrote] {out}")
    print()
    print("  汇总：跨模型相关 vs 同模型两半自比（越大越稳定）")
    for r in rows:
        print("   类 %2d %-14s  r_cross=%.3f  r_self=%.3f/%.3f  → %s"%(
            r["class"], r["name"], r["r_cross"], r["r_self_orig"], r["r_self_new"],
            "稳定" if r["r_cross"]>0.9 else "偏不稳"))


if __name__ == "__main__":
    main()
