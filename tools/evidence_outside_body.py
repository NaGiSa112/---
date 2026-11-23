"""Is class-1 evidence leaking outside the thorax? Measured without segmentation.

Why no segmentation: on portable films the air outside the body and the air inside the lungs
have similar brightness, so a border-connected flood fill swallows the lung fields (the first
version of this script did exactly that, and reported 100% "outside body" nonsense).

Robust alternative
------------------
1. Fixed peripheral bands on the 544x544 frame, where no lung can be:
     top     y < 0.12    (neck / above the apices / collimation)
     bottom  y > 0.88    (below the diaphragm / abdomen)
     sides   x < 0.06 or x > 0.94
     frame   union of the 8% border
2. For the SAME image, compute every class's evidence share inside those bands.
   The class-1 EXCESS over the image's own mean over the other nine classes is what is
   class-specific; the shared part is a generic "edge attention" artefact.
3. Report per class: mean share, and mean excess.

    python tools/evidence_outside_body.py --n 120
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

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


def bands(h: int, w: int) -> dict[str, np.ndarray]:
    yy, xx = np.mgrid[0:h, 0:w]
    yn, xn = yy / h, xx / w
    top = yn < 0.12
    bottom = yn > 0.88
    sides = (xn < 0.06) | (xn > 0.94)
    frame = (yn < 0.08) | (yn > 0.92) | (xn < 0.08) | (xn > 0.92)
    return {"top": top, "bottom": bottom, "sides": sides, "frame": frame}


def spatial(model, x):
    feat = model._spatial_features(x)
    w, b = model.head[1].weight, model.head[1].bias
    return torch.einsum("bchw,oc->bohw", feat.float(), w) + b.view(1, -1, 1, 1)


def font(size):
    for n in ("msyh.ttc", "simhei.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(n, size)
        except Exception:
            continue
    return ImageFont.load_default()


def jet(x):
    x = np.clip(x, 0, 1)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], -1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120, help="每类取多少张正样本")
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "outside_body"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr = pd.read_csv(os.path.join(ROOT, "cxr_first", "train.csv"))
    Y = np.stack([parse_label(s) for s in tr.Predict_class])
    for j, c in enumerate(range(1, 11)):
        tr[f"c{c}"] = Y[:, j]
    fold = pd.read_csv(os.path.join(ROOT, "04_新一轮", "00_规则", "patient_folds_5f.csv"))
    fmap = dict(zip(fold.Study_id.astype(str), fold.fold.astype(int)))
    models: dict[int, torch.nn.Module] = {}
    rng = np.random.default_rng(0)

    rows, panel = [], []
    print("  %-18s %8s %8s %8s %9s %9s" % ("类别", "上带", "下带", "两侧", "边框带", "相对其它类"))
    for c in range(1, 11):
        pos = tr[tr[f"c{c}"] > 0]
        pick = pos.iloc[rng.choice(len(pos), min(a.n, len(pos)), replace=False)]
        rec = {k: [] for k in ("top", "bottom", "sides", "frame")}
        excess = []
        for _, r in pick.iterrows():
            f = fmap.get(str(r.Study_id), 0)
            if f not in models:
                m, _ = load_checkpoint(os.path.join(ROOT, "runs", f"oof_f{f}", "last.pt"), device)
                m.eval()
                models[f] = m
            path = image_path("train", r.Subject_id, r.Study_id, r.Dicom_id)
            raw = load_square(path, CACHE)
            x = preprocess(raw, IMG, "per_image").unsqueeze(0).to(device)
            with torch.no_grad():
                mm = spatial(models[f], x)[0].cpu().numpy()
            mm = np.maximum(mm, 0)
            # 空间图是 544/32 = 17x17 的网格，掩膜必须按网格尺寸生成
            bd = bands(mm.shape[1], mm.shape[2])
            shares = {}
            for k, mask in bd.items():
                per_class = np.array([mm[cc][mask].sum() / (mm[cc].sum() + 1e-9)
                                      for cc in range(10)])
                shares[k] = per_class
            for k in rec:
                rec[k].append(shares[k][c - 1])
            others = np.mean([shares["frame"][i] for i in range(10) if i != c - 1])
            excess.append(shares["frame"][c - 1] - others)
            if c == 1:
                panel.append((shares["frame"][c - 1], raw, mm[c - 1], r))
        print("  %-18s %7.1f%% %7.1f%% %7.1f%% %8.1f%% %+8.1f%%" % (
            f"{c} {NAMES[c]}", 100 * np.mean(rec["top"]), 100 * np.mean(rec["bottom"]),
            100 * np.mean(rec["sides"]), 100 * np.mean(rec["frame"]),
            100 * np.mean(excess)))
        rows.append({"class": c, "name": NAMES[c],
                     "top": float(np.mean(rec["top"])), "bottom": float(np.mean(rec["bottom"])),
                     "sides": float(np.mean(rec["sides"])), "frame": float(np.mean(rec["frame"])),
                     "frame_sd": float(np.std(rec["frame"])),
                     "excess_vs_other_classes": float(np.mean(excess)), "n": len(pick)})

    pd.DataFrame(rows).to_csv(os.path.join(a.out, "band_evidence_by_class.csv"),
                              index=False, encoding="utf-8-sig")
    print()
    print("  边框带 = 宽 8% 的外框；相对其它类 = 本类边框带占比 − 同一张图上其余九类的平均")
    print("  [wrote] %s" % os.path.join(a.out, "band_evidence_by_class.csv"))

    panel.sort(key=lambda t: -t[0])
    T = 268
    W = 3 * (T + 8) + 8
    H = 44 + 2 * (T + 30 + 8) + 16
    cv = Image.new("RGB", (W, H), (250, 250, 250))
    d2 = ImageDraw.Draw(cv)
    d2.text((8, 12), "类 1：边框带证据占比最高的 6 张（白色虚线 = 8% 边框带）",
            font=font(15), fill=(20, 20, 20))
    for i, (sh, raw, heat, r) in enumerate(panel[:6]):
        gray = np.asarray(Image.fromarray(raw).resize((IMG, IMG), Image.BILINEAR), np.float32)
        hn = (heat / (heat.max() + 1e-9) * 255).astype(np.uint8)
        heat_r = np.asarray(Image.fromarray(hn).resize((IMG, IMG), Image.BILINEAR), np.float32)
        base = np.repeat(gray[..., None], 3, -1) / 255.0
        im = Image.fromarray(((0.42 * base + 0.58 * jet(heat_r / 255.0)) * 255).astype(np.uint8))
        dr = ImageDraw.Draw(im)
        m = int(round(0.08 * IMG))
        dr.rectangle([m, m, IMG - 1 - m, IMG - 1 - m], outline=(255, 255, 255))
        x0 = 8 + (i % 3) * (T + 8)
        y0 = 48 + (i // 3) * (T + 30 + 8)
        cv.paste(im.resize((T, T), Image.BILINEAR), (x0, y0))
        d2.rectangle([x0, y0, x0 + T - 1, y0 + T - 1], outline=(200, 200, 205))
        d2.text((x0, y0 + T + 4), "边框带 %.0f%%   %s" % (100 * sh, str(r.ViewPosition)[:10]),
                font=font(12), fill=(30, 30, 30))
    out = os.path.join(a.out, "class1_border_band.png")
    cv.save(out)
    print("  [wrote] %s" % out)


if __name__ == "__main__":
    main()
