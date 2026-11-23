"""Look at where the model actually looks, per class — with an honest colour scale.

WHY THE COLOUR SCALE CHANGED (2026-11-19)
-----------------------------------------
The first version normalised each image independently: `relu(M)` then min-max to 0..1.
On an image where the model finds nothing, the map is flat and slightly negative; relu keeps
only noise, and min-max then stretches that noise to full contrast, so a burned-in `PORTABLE`
label or the frame edge shows up as a confident red blob at p = 0.05.

Measured on class 1: 59% of positives have <5% of their evidence in the border band (median 0%),
while the 7.5% of images that DO put >50% there are exactly the ones with the lowest predicted
probability (0.054 vs 0.077-0.129), and in 100% of those the map peak sits in the border band.
Blank inputs (all black / grey / white) never peak in the border, and erasing the burned-in text
changes the class-1 probability by only -0.0006. So the blob is an artefact of per-image
normalisation on an empty map, not a learned shortcut.

The fix: one robust window PER CLASS, computed from the class's own response distribution
(median and 99th percentile over all sampled positives and negatives), and applied identically to
every image. A near-empty map stays blue; a real lesion still shows red. Pass --per-image-norm to
reproduce the old behaviour.

WHAT IT WRITES (reports/heatmaps/):
  class_<k>.png        mean evidence map over N positives, with a fixed per-class scale
  class_<k>_diff.png   mean positive minus mean negative — what the class actually uses
  class_<k>_img.png    the mean image, to judge anatomy alignment
  diagnostics.csv      border share, corner share, centre share, left/right balance

READ IT LIKE THIS:
  border_share  fraction of evidence within 8% of the frame edge; high means a marker or padding
  corner_share  the four 12% corner boxes
  centre_share  the middle 50% box, where lungs and heart actually are
  peak_in_border fraction of the sampled images whose map maximum sits in the border band

Usage:
    python tools/heatmap_probe.py --ckpt runs/cnxt512_pat_e12/best.pt --per-class 20
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cxr.cache import image_path, load_square  # noqa: E402
from cxr.infer import load_checkpoint, preprocess  # noqa: E402

PC = [f"p{k}" for k in range(1, 11)]


def parse_label(s: str) -> np.ndarray:
    v = np.zeros(10)
    for part in str(s).split(";"):
        part = part.strip()
        if part:
            v[int(part) - 1] = 1.0
    return v


def jet(x: np.ndarray) -> np.ndarray:
    """Minimal jet colormap so the tool needs no matplotlib."""
    x = np.clip(x, 0, 1)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def overlay(gray: np.ndarray, heat: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    base = np.repeat(gray[..., None], 3, axis=-1).astype(np.float32) / 255.0
    return (1 - alpha) * base + alpha * jet(heat)


def spatial_logits(model, x: torch.Tensor) -> torch.Tensor:
    """Per-position class logits B x 10 x h x w, the map the LSE pooling reads."""
    feat = model._spatial_features(x)
    w, b = model.head[1].weight, model.head[1].bias
    m = torch.einsum("bchw,oc->bohw", feat.float(), w) + b.view(1, -1, 1, 1)
    return m


def region_stats(heat: np.ndarray) -> dict:
    h, w = heat.shape
    yy, xx = np.mgrid[0:h, 0:w]
    yn, xn = yy / h, xx / w
    border = (yn < 0.08) | (yn > 0.92) | (xn < 0.08) | (xn > 0.92)
    corner = (((yn < 0.12) | (yn > 0.88)) & ((xn < 0.12) | (xn > 0.88)))
    centre = (yn > 0.25) & (yn < 0.75) & (xn > 0.25) & (xn < 0.75)
    tot = heat.sum() + 1e-9
    left = heat[:, : w // 2].sum()
    right = heat[:, w // 2:].sum()
    yy_p, xx_p = np.unravel_index(int(np.argmax(heat)), heat.shape)
    return {
        "border_share": float(heat[border].sum() / tot),
        "corner_share": float(heat[corner].sum() / tot),
        "centre_share": float(heat[centre].sum() / tot),
        "lr_balance": float(min(left, right) / (max(left, right) + 1e-9)),
        "peak_in_border": float(bool(border[yy_p, xx_p])),
    }


def windowed(maps: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Apply one fixed logit window to every map: below lo stays cold, above hi saturates."""
    return np.clip((maps - lo) / (hi - lo + 1e-9), 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--cache-size", type=int, default=512)
    ap.add_argument("--per-class", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "heatmaps"))
    ap.add_argument("--per-image-norm", action="store_true",
                    help="旧口径：逐图 min-max（会让空图冒出假红块，仅用于复现历史图）")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load_checkpoint(args.ckpt, device)
    if getattr(model, "lse_alpha", 0) <= 0:
        raise SystemExit("this checkpoint does not use LSE pooling, no per-position map")

    df = pd.read_csv(os.path.join(ROOT, "cxr_first", f"{args.split}.csv"))
    Y = np.stack([parse_label(s) for s in df["Predict_class"]])
    rows = []
    rng = np.random.default_rng(0)
    print(f"  ckpt={args.ckpt}  split={args.split}  per_class={args.per_class}  "
          f"norm={'per-image(旧)' if args.per_image_norm else 'per-class window'}")
    for c in range(10):
        idx = np.where(Y[:, c] > 0)[0]
        if len(idx) == 0:
            continue
        pick = rng.choice(idx, min(args.per_class, len(idx)), replace=False)
        neg_idx = np.where(Y[:, c] <= 0)[0]
        neg_pick = rng.choice(neg_idx, min(args.per_class, len(neg_idx)), replace=False)

        pos_maps, neg_maps, imgs = [], [], []
        for i in np.concatenate([pick, neg_pick]):
            r = df.iloc[i]
            path = image_path(args.split, r.Subject_id, r.Study_id, r.Dicom_id)
            raw = load_square(path, args.cache_size)
            x = preprocess(raw, args.img_size, "per_image").unsqueeze(0).to(device)
            with torch.no_grad():
                m = spatial_logits(model, x)
            mp = m[0, c].cpu().numpy()
            (pos_maps if i in set(pick.tolist()) else neg_maps).append(mp)
            if len(imgs) < len(pick):
                imgs.append(np.asarray(Image.fromarray(raw).resize(
                    (args.img_size, args.img_size), Image.BILINEAR), np.float32))

        pos_maps = np.stack(pos_maps)
        neg_maps = np.stack(neg_maps) if neg_maps else pos_maps[:0]
        # ---- 固定窗口：本类所有采样图的 50 与 99 分位 ----
        allm = np.concatenate([pos_maps.ravel(),
                               neg_maps.ravel() if len(neg_maps) else pos_maps.ravel()])
        lo, hi = float(np.percentile(allm, 50)), float(np.percentile(allm, 99.9))
        if hi - lo < 1e-6:
            hi = lo + 1.0

        def to_full(mp):
            if args.per_image_norm:
                v = np.maximum(mp, 0)
                v = v / (v.max() + 1e-9)
            else:
                v = windowed(mp, lo, hi)
            return np.asarray(Image.fromarray((v * 255).astype(np.uint8))
                              .resize((args.img_size, args.img_size), Image.BILINEAR),
                              np.float32) / 255.0

        pos_full = [to_full(mp) for mp in pos_maps]
        neg_full = [to_full(mp) for mp in neg_maps] if len(neg_maps) else [np.zeros(
            (args.img_size, args.img_size), np.float32)]
        for mp in pos_maps:
            rows.append({"class": c + 1, "kind": "pos_image", "dicom": "SAMPLE",
                         **region_stats(np.maximum(mp, 0))})
        mean_h = np.mean(pos_full, axis=0)
        mean_i = np.mean(imgs, axis=0)
        neg_h = np.mean(neg_full, axis=0)
        diff = np.clip(mean_h - neg_h, 0, None)
        diff = diff / (diff.max() + 1e-9)
        Image.fromarray((overlay(mean_i.astype(np.uint8), diff) * 255).astype(np.uint8)).save(
            os.path.join(args.out, f"class_{c + 1}_diff.png"))
        rows.append({"class": c + 1, "kind": "pos_minus_neg", "dicom": "MEAN",
                     **region_stats(diff)})
        Image.fromarray((overlay(mean_i.astype(np.uint8), mean_h) * 255).astype(np.uint8)).save(
            os.path.join(args.out, f"class_{c + 1}.png"))
        Image.fromarray(np.repeat(mean_i[..., None], 3, -1).astype(np.uint8)).save(
            os.path.join(args.out, f"class_{c + 1}_img.png"))
        peaks = np.array([region_stats(np.maximum(mp, 0))["peak_in_border"] for mp in pos_maps])
        print(f"  class {c + 1}: {len(pos_maps)} pos / {len(neg_maps)} neg, "
              f"window=[{lo:.2f}, {hi:.2f}], 峰值在边框带 {100*peaks.mean():.0f}%", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(args.out, "diagnostics_raw.csv"), index=False)
    num = [c for c in d.columns if c not in ("class", "kind", "dicom")]
    agg = d.groupby(["class", "kind"])[num].mean().reset_index()
    agg.to_csv(os.path.join(args.out, "diagnostics.csv"), index=False)
    print()
    print("evidence distribution by class and by map type:")
    for kind in agg["kind"].unique():
        sub = agg[agg["kind"] == kind].drop(columns=["kind"])
        print(f"  [{kind}]")
        print(sub.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()
