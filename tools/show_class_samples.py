"""Visual inspection: what do classes 1 / 4 / 8 look like, hits vs misses?

Uses the five-fold OOF predictions (single model, no-KD, fixed last12) so the sample pool is
the whole training set rather than the 24 class-1 positives in official valid.

For each class it shows two rows:
    命中   该类的真阳性里，模型给分最高的几张
    漏检   该类的真阳性里，模型给分最低的几张

Each tile = the image + the model's own per-class evidence map (the LSE spatial logits for
that class), with the study's full label string and the predicted probability.

    python tools/show_class_samples.py --classes 1,4,8 --per-row 4
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

NAMES = {1: "Enlarged Cardiomediastinum", 2: "Pneumothorax", 3: "Consolidation",
         4: "Pneumonia", 5: "Edema", 6: "Cardiomegaly", 7: "Atelectasis",
         8: "Lung Opacity", 9: "Pleural Effusion", 10: "No Finding"}
IMG = 544
CACHE = 512
TILE = 236
PAD = 8
CAP = 34


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


def font(size: int):
    for n in ("msyh.ttc", "msyhbd.ttc", "simhei.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(n, size)
        except Exception:
            continue
    return ImageFont.load_default()


def jet(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def spatial(model, x):
    feat = model._spatial_features(x)
    w, b = model.head[1].weight, model.head[1].bias
    return torch.einsum("bchw,oc->bohw", feat.float(), w) + b.view(1, -1, 1, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", default="1,4,8")
    ap.add_argument("--per-row", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "samples_class_1_4_8.png"))
    a = ap.parse_args()
    classes = [int(c) for c in a.classes.split(",") if c.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 标签与 OOF 预测 ----
    # 标签是**逐图**的：每张图有自己的 Predict_class。按 study 取 max 会把阴性图也算成阳性。
    # 这里改为图像级：正样本 = 该图自身标签含该类；预测值取它所属 study 的 OOF 概率。
    tr = pd.read_csv(os.path.join(ROOT, "cxr_first", "train.csv"))
    Y = np.stack([parse_label(s) for s in tr.Predict_class])
    img = tr[["Subject_id", "Study_id", "Dicom_id", "ViewPosition"]].copy()
    for j, c in enumerate(range(1, 11)):
        img[c] = Y[:, j]
    img["labstr"] = tr.Predict_class.astype(str).values

    parts = [pd.read_csv(os.path.join(ROOT, "runs", f"oof_f{f}_study.csv")) for f in range(5)]
    pred = pd.concat(parts)
    pred = pred.drop_duplicates(subset=["Subject_id", "Study_id", "Predict_class"])
    wide = pred.pivot_table(index=["Subject_id", "Study_id"], columns="Predict_class",
                            values="Probability", aggfunc="first").reindex(columns=list(range(1, 11)))
    # 把 study 级概率贴到每张图上
    img = img.set_index(["Subject_id", "Study_id"]).join(wide, how="inner",
                                                           rsuffix="_pred").reset_index()
    # join 会把整数列名转成字符串，这里统一映射
    LBL = {c: str(c) for c in range(1, 11)}
    PRED = {c: f"{c}_pred" for c in range(1, 11)}

    fold_of = pd.read_csv(os.path.join(ROOT, "04_新一轮", "00_规则", "patient_folds_5f.csv"))
    fold_map = dict(zip(fold_of.Study_id.astype(str), fold_of.fold.astype(int)))

    fnt = font(15)
    small = font(12)
    cols = a.per_row
    W = PAD + cols * (TILE + PAD)
    H = PAD + len(classes) * (30 + 2 * (TILE + CAP + PAD)) + 30
    canvas = Image.new("RGB", (W, H), (252, 252, 252))
    d = ImageDraw.Draw(canvas)
    d.text((PAD, 8), "类别 1 / 4 / 8 样本目视：命中 vs 漏检（图像 + 模型自身证据图）",
           font=fnt, fill=(20, 20, 20))

    y = PAD + 24
    rows_log = []
    models: dict[int, torch.nn.Module] = {}
    for c in classes:
        pos = img[img[LBL[c]] > 0]
        pc = PRED[c]
        hits = pos.sort_values(pc, ascending=False).head(a.per_row)
        miss = pos.sort_values(pc, ascending=True).head(a.per_row)
        # 固定窗口：用该类 60 张正样本的 p50/p99 作为统一色标（避免逐图归一化造假红块）
        cal = pos.iloc[np.random.default_rng(2).choice(len(pos), min(60, len(pos)), replace=False)]
        cal_maps = []
        for _, rr in cal.iterrows():
            ff = fold_map.get(str(rr.Study_id), 0)
            if ff not in models:
                mm_, _ = load_checkpoint(os.path.join(ROOT, "runs", f"oof_f{ff}", "last.pt"), device)
                mm_.eval()
                models[ff] = mm_
            rawc = load_square(image_path("train", rr.Subject_id, rr.Study_id, rr.Dicom_id), CACHE)
            xc = preprocess(rawc, IMG, "per_image").unsqueeze(0).to(device)
            with torch.no_grad():
                cal_maps.append(spatial(models[ff], xc)[0, c - 1].cpu().numpy())
        flat = np.concatenate([m.ravel() for m in cal_maps])
        WLO, WHI = float(np.percentile(flat, 50)), float(np.percentile(flat, 99.9))
        if WHI - WLO < 1e-6:
            WHI = WLO + 1.0
        d.rectangle([0, y, W, y + 26], fill=(238, 240, 248))
        d.text((PAD, y + 5), f"类 {c}  {NAMES[c]}   （图像级真阳性 {len(pos)} 张）",
               font=fnt, fill=(30, 30, 90))
        y += 30
        for tag, sel in (("命中（模型高分）", hits), ("漏检（模型低分）", miss)):
            d.text((PAD, y), tag, font=small, fill=(120, 60, 60) if "漏" in tag else (60, 100, 60))
            y += 16
            for k, (_, row) in enumerate(sel.iterrows()):
                study = row.Study_id
                f = fold_map.get(study, 0)
                if f not in models:
                    ck = os.path.join(ROOT, "runs", f"oof_f{f}", "last.pt")
                    m, _ = load_checkpoint(ck, device)
                    m.eval()
                    models[f] = m
                path = image_path("train", row.Subject_id, study, row.Dicom_id)
                raw = load_square(path, CACHE)
                x = preprocess(raw, IMG, "per_image").unsqueeze(0).to(device)
                with torch.no_grad():
                    mm = spatial(models[f], x)
                heat = mm[0, c - 1].cpu().numpy()
                heat = np.clip((heat - WLO) / (WHI - WLO + 1e-9), 0, 1)   # 固定窗口
                heat = np.asarray(Image.fromarray((heat * 255).astype(np.uint8))
                                  .resize((IMG, IMG), Image.BILINEAR), np.float32) / 255.0
                gray = np.asarray(Image.fromarray(raw).resize((IMG, IMG), Image.BILINEAR), np.float32)
                base = np.repeat(gray[..., None], 3, -1) / 255.0
                rgb = (0.45 * base + 0.55 * jet(heat))
                x0 = PAD + k * (TILE + PAD)
                canvas.paste(Image.fromarray((rgb * 255).astype(np.uint8)).resize((TILE, TILE),
                                                                                 Image.BILINEAR),
                             (x0, y))
                d.rectangle([x0, y, x0 + TILE - 1, y + TILE - 1], outline=(205, 205, 210))
                d.text((x0, y + TILE + 3), f"p={float(row[pc]):.3f}  {row.ViewPosition}",
                       font=small, fill=(20, 20, 20))
                d.text((x0, y + TILE + 16), f"本图标签 {row.labstr[:16]}",
                       font=small, fill=(90, 90, 90))
                rows_log.append({"class": c, "kind": "hit" if "命中" in tag else "miss",
                                 "subject": row.Subject_id, "study": study,
                                 "dicom": row.Dicom_id, "view": row.ViewPosition,
                                 "p": float(row[pc]), "labels": row.labstr})
            y += TILE + CAP + PAD

    d.text((PAD, H - 22), "证据图为该类在 LSE 池化前的空间 logits（relu 后归一化）；红=高响应",
           font=small, fill=(100, 100, 100))
    canvas.save(a.out)
    pd.DataFrame(rows_log).to_csv(a.out.replace(".png", ".csv"), index=False, encoding="utf-8-sig")
    print(f"  [wrote] {a.out}  {W}x{H}")
    print(f"  [wrote] {a.out.replace('.png', '.csv')}")
    print()
    for c in classes:
        sub = [r for r in rows_log if r["class"] == c]
        h = [r["p"] for r in sub if r["kind"] == "hit"]
        m = [r["p"] for r in sub if r["kind"] == "miss"]
        print(f"  类 {c}: 命中 p={min(h):.3f}–{max(h):.3f}   漏检 p={min(m):.3f}–{max(m):.3f}")


if __name__ == "__main__":
    main()
