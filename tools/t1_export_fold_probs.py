"""T1 交付物：把 last.pt 导出成 fold2 留出集的逐图概率 + 检查级指标自检。

口径与 `src/cxr/infer.py` 的部署路径一致：
- 预处理：JPEG → draft 0.98×cache_size → BILINEAR 到 img_size → 逐图标准化；
  与训练时读 512 缓存再缩到 384 的输入逐像素同源（同一个 load_square）。
- TTA：水平翻转，两次前向的概率取平均（不是 logits 平均）。
- 检查级：侧位 0.6 权重加权均值；患者收缩 0.85 单独报告，不改 CSV 本身。

    python tools/t1_export_fold_probs.py --ckpt runs_t1/evb1_384_f2/last.pt \
        --fold 2 --img 384 --cache-size 512 --out runs_t1/evb1_384_f2/fold2_probs.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cxr.infer import PathDataset, predict_probs  # noqa: E402
from cxr.models import build_model  # noqa: E402

PCOLS = [f"p{k}" for k in range(1, 11)]


def parse_label(s) -> np.ndarray:
    """与 tools/study_metric.py 逐行一致：只有整串恰为 "10" 时类 10 才算阳性。"""
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


def rank_auc(y: np.ndarray, p: np.ndarray) -> float:
    """与 tools/study_metric.py 同款秩统计 AUC（并列取平均秩）。"""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    order = np.argsort(p, kind="mergesort")
    ps = p[order]
    r = np.empty(len(p), float)
    # 平均秩处理并列，避免分数离散时 AUC 被高估
    i = 0
    while i < len(ps):
        j = i
        while j + 1 < len(ps) and ps[j + 1] == ps[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    pos = y > 0
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def macro_auc(Y: np.ndarray, P: np.ndarray):
    per = [rank_auc(Y[:, c], P[:, c]) for c in range(Y.shape[1])]
    return float(np.nanmean(per)), per


def study_level(df: pd.DataFrame, probs: np.ndarray, lat_weight: float = 0.6,
                shrink: float = 0.85):
    """逐图 -> 检查级（侧位加权）-> 患者收缩，完全复刻 infer.write_submission。"""
    d = pd.DataFrame(probs, columns=PCOLS)
    d["Subject_id"] = df["Subject_id"].values
    d["Study_id"] = df["Study_id"].values
    w = np.where(df["ViewPosition"].fillna("NA").astype(str).isin(["LATERAL", "LL"]).values,
                 lat_weight, 1.0)
    for c in PCOLS:
        d[c] = d[c] * w
    g = d.groupby(["Subject_id", "Study_id"], as_index=False)[PCOLS].sum()
    wsum = pd.Series(w).groupby([df["Subject_id"].values, df["Study_id"].values]).sum()
    g[PCOLS] = g[PCOLS].div(wsum.values[:, None])
    # 检查级标签：同一检查的所有图标签一致（已在 docs 中验证），取首行
    lab = (df.assign(k=df.Study_id.astype(str))
             .drop_duplicates("k")[["Study_id", "Predict_class"]])
    g = g.merge(lab, on="Study_id", how="left")
    Y = np.stack([parse_label(v) for v in g["Predict_class"].astype(str)])
    P = g[PCOLS].values.astype(np.float32)
    P_shrunk = P.copy()
    pm = g.groupby("Subject_id")[PCOLS].transform("mean").values
    P_shrunk = shrink * P + (1.0 - shrink) * pm
    return g, Y, P, P_shrunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="fold", choices=["fold", "valid"],
                    help="fold=训练折表里的留出折；valid=官方 valid 1422 图")
    ap.add_argument("--fold", type=int, default=2)
    ap.add_argument("--img", type=int, default=0, help="0=读 ckpt 里的 img_size")
    ap.add_argument("--cache-size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", {})
    arch = cfg.get("arch") or ck.get("arch")
    img = a.img or int(cfg.get("img_size") or ck.get("img_size") or 384)
    in_ch = 2 if cfg.get("extra_channel", "") == "asym" else 1
    print(f"[export] ckpt={a.ckpt}\n[export] arch={arch} img={img} cache={a.cache_size} "
          f"epoch={ck.get('epoch')} lse_alpha={cfg.get('lse_alpha')} head={cfg.get('head_type')}",
          flush=True)

    model = build_model(arch, pretrained=False, drop_path=0.0,
                        n_aux=0, n_view=0, final_norm=bool(cfg.get("final_norm", False)),
                        lse_alpha=float(cfg.get("lse_alpha", 0.0)),
                        lse_tau=float(cfg.get("lse_tau", 1.0)),
                        head_type=cfg.get("head_type", "lse"), in_ch=in_ch, img_size=0)
    sd = ck["model"] if "model" in ck else ck
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected or [k for k in missing if not k.endswith("num_batches_tracked")]:
        raise SystemExit(f"[export] state dict 不匹配 missing={missing[:5]} unexpected={unexpected[:5]}")
    model.eval().cuda()
    print(f"[export] loaded {sum(p.numel() for p in model.parameters())/1e6:.3f}M params", flush=True)

    data_dir = os.environ.get("CXR_DATA", "")
    tr = pd.read_csv(os.path.join(data_dir, "train.csv"))
    if a.split == "valid":
        vd = pd.read_csv(os.path.join(data_dir, "valid.csv"))
        df = vd.reset_index(drop=True)
        split_name = "valid"
        print(f"[export] valid: {len(df)} images / {df.Study_id.nunique()} studies / "
              f"{df.Subject_id.nunique()} patients", flush=True)
    else:
        fk = pd.read_csv(os.path.join(ROOT, "data", "patient_folds_5f.csv"))
        fmap = dict(zip(fk.Study_id.astype(str), fk.fold.astype(int)))
        fo = tr.Study_id.astype(str).map(fmap)
        df = tr[(fo == a.fold).values].reset_index(drop=True)
        split_name = f"fold{a.fold}"
        print(f"[export] fold {a.fold}: {len(df)} images / {df.Study_id.nunique()} studies / "
              f"{df.Subject_id.nunique()} patients", flush=True)

    ds = PathDataset(df, split_name if a.split == "valid" else "train",
                     img_size=img, cache_size=a.cache_size, norm="per_image")
    probs = predict_probs(model, ds, device="cuda", batch_size=a.batch_size, tta=True, amp=a.amp)
    print(f"[export] probs {probs.shape} finite={np.isfinite(probs).all()} "
          f"range=[{probs.min():.4f},{probs.max():.4f}]", flush=True)

    out = pd.DataFrame({"Dicom_id": df.Dicom_id.values, "Subject_id": df.Subject_id.values,
                        "Study_id": df.Study_id.values})
    for k, c in enumerate(PCOLS):
        out[c] = probs[:, k]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out.to_csv(a.out, index=False, float_format="%.6f")
    print(f"[export] wrote {a.out}  rows={len(out)}", flush=True)

    # ---- 指标自检 ----
    Y_img = np.stack([parse_label(v) for v in df["Predict_class"].astype(str)])
    m_img, per_img = macro_auc(Y_img, probs)
    g, Y, P, P_sh = study_level(df, probs)
    m_st, per_st = macro_auc(Y, P)
    m_sh, per_sh = macro_auc(Y, P_sh)
    print(f"[metric] image-level macro   {m_img:.4f}")
    print(f"[metric] study-level macro   {m_st:.4f}   (侧位加权 0.6，检查级)")
    print(f"[metric] study + shrink 0.85 {m_sh:.4f}")
    print(f"[metric] studies {len(g)}  c1+ {int(Y[:,0].sum())}  c4+ {int(Y[:,3].sum())}")
    print("[metric] per-class study-level: " +
          " ".join(f"c{i+1}:{v:.3f}" for i, v in enumerate(per_st)))

    rec = dict(ckpt=os.path.abspath(a.ckpt), arch=arch, img_size=img,
               cache_size=a.cache_size, split=split_name, n_images=int(len(df)),
               n_studies=int(len(g)), grayscale_note="stem 灰度化，单通道",
               tta="hflip, 概率平均", macro_image=m_img, macro_study=m_st,
               macro_study_shrink085=m_sh, per_class_study=per_st,
               c1_pos=int(Y[:, 0].sum()), c4_pos=int(Y[:, 3].sum()))
    js = a.out.replace(".csv", "_metrics.json")
    with open(js, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    print(f"[metric] wrote {js}", flush=True)


if __name__ == "__main__":
    main()
