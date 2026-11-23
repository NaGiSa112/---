"""类 1 冻结表征探针：DINOv2 CLS  vs  Tiny 全局特征，同协议对照。

设计（对应对抗审阅 §5、§9 第 3–6 步）
------------------------------------
折：探索 **fold 2**（其余四折训练），复核留 fold 3。
画布：448×448（14 | 448、32 | 448，两臂同视野、同解码、同归一化）。
两臂：
    A = 现用 Tiny（oof_f2/last.pt）分类器前的**全局平均**特征，768 维
    B = 通用 DINOv2 ViT-B/14（LVD-142M，无 register）最后 block 的 CLS，768 维
    R = 现用 Tiny 原头（带 LSE 空间池化）在该折上的实际分数，作锚
头：768→10 线性层（7,690 参数），两臂同初始化、同批次顺序、同预算；
标签：**Study 级硬 0/1**（图 sigmoid → 侧位 0.6 加权均值 → 检查级 BCE）。
指标：检查级 AUC，主看**类 1**（纵隔心影增大），另报十类与宏。

    python tools/dinov2_probe.py --extract        # 提特征并缓存（约 7 分钟 GPU）
    python tools/dinov2_probe.py --fit            # 训练头并出结果（CPU/GPU 都很快）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

ROOT = r"E:\挑战赛"
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cxr.cache import image_path, load_square, open_cache  # noqa: E402
from cxr.infer import load_checkpoint                    # noqa: E402
from dinov2_features import load_dinov2                  # noqa: E402

PCOLS = list(range(1, 11))
CACHE = os.path.join(ROOT, "runs", "probe448")
CANVAS = 448
FOLDS = os.path.join(ROOT, "04_新一轮", "00_规则", "patient_folds_5f.csv")

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


def prepare_img(raw: np.ndarray):
    """448 画布、per-image 标准化；返回 (1ch, 3ch) 两个张量。"""
    g = np.asarray(raw, np.float32)
    g = (g - float(g.mean())) / max(float(g.std()), 1e-3)
    t1 = torch.from_numpy(g.astype(np.float32))[None]
    t3 = torch.from_numpy(np.stack([g, g, g]).astype(np.float32))
    return t1, t3


def extract() -> None:
    os.makedirs(CACHE, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr = pd.read_csv(os.path.join(ROOT, "cxr_first", "train.csv"))
    fold = pd.read_csv(FOLDS)
    fmap = dict(zip(fold.Study_id.astype(str), fold.fold.astype(int)))

    mm, meta_list = open_cache("train", 512)          # memmap，按行索引，避免 3 万次 JPEG 解码
    row_of = {d: i for i, d in enumerate(meta_list)}
    tiny, _ = load_checkpoint(os.path.join(ROOT, "runs", "oof_f2", "last.pt"), device)
    tiny.eval()
    dino = load_dinov2(img=CANVAS).to(device).eval()

    fa_list, fb_list, meta = [], [], []
    bs = 24
    t0 = time.time()
    for start in range(0, len(tr), bs):
        chunk = tr.iloc[start:start + bs]
        t1s, t3s = [], []
        for _, r in chunk.iterrows():
            # 直接读 512 缓存行再缩到 448（等价视野，省掉 3 万次 JPEG 解码）
            idx_row = row_of.get(r.Dicom_id)
            if idx_row is None:
                raw = load_square(image_path("train", r.Subject_id, r.Study_id, r.Dicom_id), 512)
            else:
                raw = np.asarray(mm[idx_row])
            raw = np.asarray(Image.fromarray(raw.astype(np.uint8))
                             .resize((CANVAS, CANVAS), Image.BILINEAR))
            a, b = prepare_img(raw)
            t1s.append(a)
            t3s.append(b)
            meta.append({"Subject_id": r.Subject_id, "Study_id": r.Study_id,
                         "Dicom_id": r.Dicom_id, "ViewPosition": r.ViewPosition})
        x1 = torch.stack(t1s).to(device)
        x3 = torch.stack(t3s).to(device)
        with torch.no_grad():
            f = tiny._spatial_features(x1)                 # B,768,14,14
            fa = f.mean(dim=(2, 3))                        # 全局平均 → 768
            fb = dino(x3)                                  # CLS → 768
        fa_list.append(fa.float().cpu().numpy().astype(np.float16))
        fb_list.append(fb.float().cpu().numpy().astype(np.float16))
        if (start // bs) % 100 == 0:
            el = time.time() - t0
            print(f"    {start+len(chunk)}/{len(tr)}  用时 {el/60:.1f} 分  "
                  f"预计剩余 {el/(start+len(chunk))*(len(tr)-start-len(chunk))/60:.1f} 分", flush=True)

    np.save(os.path.join(CACHE, "tiny_global.npy"), np.concatenate(fa_list))
    np.save(os.path.join(CACHE, "dino_cls.npy"), np.concatenate(fb_list))
    pd.DataFrame(meta).assign(fold=[fmap.get(str(s), -1) for s in [m["Study_id"] for m in meta]]) \
        .to_csv(os.path.join(CACHE, "index.csv"), index=False)
    print(f"  [wrote] {os.path.relpath(CACHE, ROOT)}/  "
          f"tiny_global.npy / dino_cls.npy / index.csv  用时 {(time.time()-t0)/60:.1f} 分")


def fit_head(F_tr, y_tr, n_epochs=12, seed=20261120, lr=1e-3, wd=1e-4, bs=64):
    """Study 级 BCE 训练一个 768→10 线性头。F_tr/y_tr 已按 study 聚合完成。"""
    torch.manual_seed(seed)
    head = nn.Linear(F_tr.shape[1], 10)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss()
    n = len(F_tr)
    for ep in range(n_epochs):
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed + ep))
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            out = head(F_tr[idx])
            loss = lossf(out, y_tr[idx])
            loss.backward()
            opt.step()
    return head


def fit() -> None:
    idx = pd.read_csv(os.path.join(CACHE, "index.csv"))
    FA = np.load(os.path.join(CACHE, "tiny_global.npy")).astype(np.float32)
    FB = np.load(os.path.join(CACHE, "dino_cls.npy")).astype(np.float32)
    tr = pd.read_csv(os.path.join(ROOT, "cxr_first", "train.csv"))
    Y = np.stack([parse_label(s) for s in tr.Predict_class]).astype(np.float32)
    idx["y"] = list(Y)
    # study 级标签（任一图阳性即阳性）
    st = idx.groupby(["Subject_id", "Study_id"]).agg(
        y=("y", lambda s: np.max(np.stack(list(s)), axis=0)),
        fold=("fold", "first"), n_views=("Dicom_id", "size")).reset_index()

    def pool_study(feat):
        """把图像特征按 study 聚合：侧位 0.6、其余 1.0 的加权均值（在特征层）。"""
        idx2 = idx[["Subject_id", "Study_id", "ViewPosition"]].reset_index(drop=True)
        w = np.where(idx2.ViewPosition.fillna("NA").astype(str).str.upper()
                     .isin(["LATERAL", "LL"]), 0.6, 1.0).astype(np.float32)
        out, keys = [], []
        for (s, stid), g in idx2.groupby(["Subject_id", "Study_id"]):
            rows = g.index.to_numpy()
            ww = w[rows][:, None]
            out.append((feat[rows] * ww).sum(0) / ww.sum())
            keys.append((s, stid))
        return np.stack(out).astype(np.float32), keys

    SA, keysA = pool_study(FA)
    SB, keysB = pool_study(FB)
    st = st.set_index(["Subject_id", "Study_id"]).loc[keysA].reset_index()

    hold = st.fold.to_numpy() == 2
    print(f"  样本：{int((~hold).sum())} 训练 / {int(hold.sum())} 留出（fold 2）")
    ymat = np.stack(st.y.to_numpy()).astype(np.float32)
    res = {}
    for name, S in (("A Tiny 全局特征", SA), ("B DINOv2 CLS", SB)):
        # 逐维标准化只用训练侧
        mu = S[~hold].mean(0, keepdims=True)
        sd = S[~hold].std(0, keepdims=True) + 1e-6
        Z = (S - mu) / sd
        Ft = torch.from_numpy(Z[~hold])
        yt = torch.from_numpy(ymat[~hold])
        head = fit_head(Ft, yt)
        with torch.no_grad():
            pred = torch.sigmoid(head(torch.from_numpy(Z[hold]))).numpy()
        yh = ymat[hold]
        per = [auc(yh[:, c], pred[:, c]) for c in range(10)]
        res[name] = {"per": per, "macro": float(np.nanmean(per))}
        print(f"  {name}: 类1 {per[0]:.4f}  宏 {np.nanmean(per):.4f}")
        print("     十类: " + " ".join(f"{v:.3f}" for v in per))

    # R：现用原头（带 LSE）在该折的实际分数
    p = os.path.join(ROOT, "runs", "oof_f2_study.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        w = d.pivot_table(index=["Subject_id", "Study_id"], columns="Predict_class",
                          values="Probability", aggfunc="first").reindex(columns=PCOLS)
        a0 = 0.85
        subj = w.index.get_level_values(0)
        g = (w - (1 - a0) * w.groupby(subj).transform("mean")) / a0   # 反解到收缩前
        common = g.index.intersection(st.set_index(["Subject_id", "Study_id"]).index)
        yl = st.set_index(["Subject_id", "Study_id"]).loc[common].y
        ymat2 = np.stack(yl.to_numpy())
        per = [auc(ymat2[:, c], g.loc[common, c + 1].to_numpy()) for c in range(10)]
        res["R 现用原头(LSE)"] = {"per": per, "macro": float(np.nanmean(per))}
        print(f"  R 现用原头(LSE): 类1 {per[0]:.4f}  宏 {np.nanmean(per):.4f}")

    out = os.path.join(ROOT, "reports", "dinov2_probe_2026-11-21.json")
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    d1 = res["B DINOv2 CLS"]["per"][0] - res["A Tiny 全局特征"]["per"][0]
    print()
    print(f"  主指标 Δ1 = AUC(B) − AUC(A) = {d1:+.4f}   （门槛 +0.0150）")
    print(f"  {'✅ 过门，可做 fold 3 复核' if d1 >= 0.015 else '❌ 未过门，按预案收口'}")
    print(f"  [wrote] {os.path.relpath(out, ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--fit", action="store_true")
    a = ap.parse_args()
    if a.extract:
        extract()
    if a.fit:
        fit()
    if not (a.extract or a.fit):
        print(__doc__)


if __name__ == "__main__":
    main()
