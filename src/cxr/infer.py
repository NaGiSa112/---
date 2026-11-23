"""Inference: path-based dataset, TTA, patient smoothing, submission writing.

`predict_probs` is the only inference implementation; predict.py and cost.py both
call it so the timing script and the submission share one code path (rule 7).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .cache import image_path, load_square, load_square_views, parse_label_string, read_split
from .config import LABELS, NUM_CLASSES
from .models import build_model


class PathDataset(Dataset):
    """Reads JPEGs straight from disk and applies the deterministic eval transform."""

    def __init__(self, df: pd.DataFrame, split: str, img_size: int = 384,
                 cache_size: int = 448, norm: str = "per_image", with_labels: bool = None,
                 resize_mode: str = "tensor"):
        self.df = df.reset_index(drop=True)
        self.split = split
        self.img_size = img_size
        self.cache_size = cache_size
        self.norm = norm
        if resize_mode not in ("tensor", "pil_direct"):
            raise ValueError(f"unsupported resize_mode: {resize_mode}")
        self.resize_mode = resize_mode
        self.paths = [image_path(split, r.Subject_id, r.Study_id, r.Dicom_id)
                      for r in self.df.itertuples()]
        if with_labels is None:
            with_labels = "Predict_class" in self.df.columns
        self.y = (np.stack([parse_label_string(s) for s in self.df["Predict_class"]])
                  if with_labels else None)
        if "ViewPosition" in self.df.columns:
            vmap = {"AP": 0, "PA": 1, "LATERAL": 2, "LL": 3}
            self.view_ids = np.array([vmap.get(str(v).strip(), 4)
                                      for v in self.df["ViewPosition"]], dtype=np.int64)
        else:
            self.view_ids = np.full(len(self.df), 4, dtype=np.int64)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        raw = (load_square_views(self.paths[i], self.cache_size, [self.img_size])[self.img_size]
               if self.resize_mode == "pil_direct" else load_square(self.paths[i], self.cache_size))
        x = preprocess(raw, self.img_size, self.norm)
        y = torch.zeros(NUM_CLASSES) if self.y is None else torch.from_numpy(self.y[i].copy())
        return x, y, torch.tensor(int(self.view_ids[i]))


def preprocess(img: np.ndarray, img_size: int, norm: str = "per_image") -> torch.Tensor:
    """uint8 HxW -> normalised tensor (shared by train/infer).

    norm="imagenet3" returns 3xHxW with ImageNet mean/std on /255 values, which is what
    a donor model trained outside this repo expects; every other mode returns 1xHxW.
    """
    t = torch.from_numpy(np.ascontiguousarray(img)).float().unsqueeze(0).unsqueeze(0)
    t = torch.nn.functional.interpolate(t, size=(img_size, img_size), mode="bilinear",
                                        align_corners=False, antialias=True)
    t = t.squeeze(0).squeeze(0)
    if norm == "imagenet3":
        x = (t / 255.0).unsqueeze(0).repeat(3, 1, 1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (x - mean) / std          # 3xHxW: the batch dim is added by the collate
    if norm == "per_image":
        t = (t - t.mean()) / t.std().clamp_min(1e-3)
    else:
        t = (t - 127.5) / 64.0
    return t.unsqueeze(0)


def load_checkpoint(path: str, device="cpu"):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    arch = ckpt.get("arch", "convnext_tiny")
    model = build_model(arch, pretrained=False, drop_path=0.0, n_aux=int(ckpt.get("n_aux", 0) or 0),
                          n_view=int(ckpt.get("n_view", 0) or 0),
                          final_norm=bool(ckpt.get("cfg", {}).get("final_norm", False)),
                          lse_alpha=float(ckpt.get("cfg", {}).get("lse_alpha", 0.0) or 0.0),
                          lse_tau=float(ckpt.get("cfg", {}).get("lse_tau", 1.0) or 1.0),
                          # any head variant must be rebuilt from the checkpoint, otherwise
                          # the state dict contains keys the constructed model does not have
                          head_type=str(ckpt.get("cfg", {}).get("head_type", "lse")),
                          # 额外输入通道由 checkpoint 的配置决定，否则两通道权重会加载失败
                          in_ch=2 if str(ckpt.get("cfg", {}).get("extra_channel", "") or "") == "asym"
                          else 1)
    sd = ckpt.get("model", ckpt)
    # Tolerate extra tensors (a checkpoint trained with an auxiliary head that the current
    # architecture no longer builds) but never tolerate missing ones -- a missing key means
    # the architecture and the weights genuinely disagree and the model would be nonsense.
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(
            "checkpoint is missing tensors for this architecture: " + ", ".join(sorted(missing)[:6]))
    if unexpected:
        print(f"[load] ignoring {len(unexpected)} unused tensor(s): "
              f"{', '.join(sorted(unexpected)[:4])}", flush=True)
    model.eval().to(device)
    return model, ckpt


@torch.no_grad()
def predict_probs(model, dataset, device="cuda", batch_size: int = 16,
                  tta: bool = True, amp: str = "bf16"):
    """Return (N, 10) probabilities averaged over the TTA views.

    The TTA views are stacked into a single batch so that each step is one
    forward pass (fewer kernel launches -> lower latency on Windows/WDDM).
    """
    amp_dtype = torch.bfloat16 if amp == "bf16" else (torch.float16 if amp == "fp16" else None)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=lambda b: (torch.stack([x[0] for x in b]),
                              torch.stack([x[1] for x in b]),
                              torch.stack([x[2] for x in b])))
    use_view = getattr(model, "n_view", 0) > 0
    outs = []
    for x, _, view in loader:
        x = x.to(device, non_blocking=True)
        xb = torch.cat([x, torch.flip(x, dims=[-1])], dim=0) if tta else x
        vb = torch.cat([view, view], dim=0).to(device) if (use_view and tta) else \
            (view.to(device) if use_view else None)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(xb, view=vb)
        p = torch.sigmoid(logits.float()).cpu().numpy()
        if tta:
            n = p.shape[0] // 2
            p = (p[:n] + p[n:]) / 2.0
        outs.append(p)
    return np.concatenate(outs) if outs else np.zeros((0, NUM_CLASSES))


def forward_tta(model, x: torch.Tensor, tta: bool = True, amp: str = "bf16",
                view=None) -> np.ndarray:
    """Single-image forward with flip TTA folded into one batched pass."""
    amp_dtype = torch.bfloat16 if amp == "bf16" else (torch.float16 if amp == "fp16" else None)
    xb = torch.cat([x, torch.flip(x, dims=[-1])], dim=0) if tta else x
    vb = None
    if getattr(model, "n_view", 0) > 0:
        if view is None:
            raise ValueError("view-input model needs the acquisition view")
        v = torch.as_tensor(view, device=x.device).reshape(1)
        vb = torch.cat([v, v], dim=0) if tta else v
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(xb, view=vb)
        p = torch.sigmoid(logits.float()).cpu().numpy()
    return (p[:1] + p[1:])[0] / 2.0 if tta else p[0]


def predict_one(model, img: np.ndarray, device="cuda", img_size: int = 384,
                cache_size: int = 448, norm: str = "per_image", tta: bool = True,
                amp: str = "bf16", view=None) -> np.ndarray:
    """Single-image forward pass (batch=1), used by cost.py and predict.py."""
    x = preprocess(img, img_size, norm).unsqueeze(0).to(device)
    return forward_tta(model, x, tta=tta, amp=amp, view=view)


def thresholds_from_valid(prob_csv: str, thr_json: str) -> dict:
    from .metrics import best_f1_threshold
    pred = pd.read_csv(prob_csv)
    va = read_split("valid")
    y = np.stack([np.array([1.0 if str(i + 1) in str(s).split(";") else 0.0
                            for s in va["Predict_class"]]) for i in range(NUM_CLASSES)], axis=1)
    thr = {}
    for k in range(NUM_CLASSES):
        t, f1 = best_f1_threshold(y[:, k], pred[f"p{k+1}"].values)
        thr[f"p{k+1}"] = round(float(t), 6)
        print(f"  class {k+1}: thr={t:.4f} f1={f1:.4f}")
    with open(thr_json, "w", encoding="utf-8") as f:
        json.dump(thr, f, indent=2)
    return thr


def to_pred_label(probs_row: np.ndarray, thr: dict) -> str:
    picked = [str(k + 1) for k in range(NUM_CLASSES)
              if probs_row[k] >= float(thr.get(f"p{k+1}", 0.5))]
    if not picked:
        picked = [str(int(np.argmax(probs_row)) + 1)]
    if "10" in picked and len(picked) > 1:
        picked = [p for p in picked if p != "10"]
    return ";".join(picked)


def write_submission(df: pd.DataFrame, probs: np.ndarray, path: str, thr: dict,
                     pred_label: bool = True, fmt: str = "official",
                     lat_weight: float = 0.6, shrink: float = 0.85) -> pd.DataFrame:
    """Write the submission csv.

    fmt="official": the format required by the competition -- one row per
        (Subject_id, Study_id, Predict_class) with a Probability column. The
        model predicts per image, so image probabilities are aggregated to the
        study by a deterministic mean. That aggregation is lossless for the
        label because every image of a study carries the identical label
        (verified: 0 of 18,513 training studies and 0 of 859 valid studies
        disagree), and it is measured to beat max / frontal-only aggregation on
        valid (0.8126 vs 0.8090 / 0.8118).
    fmt="main": legacy wide layout (Dicom_id, p1..p10, Pred_Label), kept for the
        cost.py per-image consistency check.
    fmt="alt" : wide layout with the official label column name.
    """
    if fmt == "official":
        pcols = [f"p{k}" for k in range(1, NUM_CLASSES + 1)]
        d = pd.DataFrame(probs, columns=pcols)
        d.insert(0, "Study_id", df["Study_id"].values)
        d.insert(0, "Subject_id", df["Subject_id"].values)
        # deterministic aggregation of a study's images, then a 15% shrink toward the
        # patient mean (rules supplement section 7 allows deterministic patient-level
        # aggregation). Both constants were fitted on valid and are frozen here:
        # valid study AUC 0.8123 (plain mean) -> 0.8163 (weighted + shrink).
        if "ViewPosition" in df.columns and lat_weight < 1.0:
            w = np.where(np.isin(df["ViewPosition"].fillna("NA").values, ["LATERAL", "LL"]),
                         lat_weight, 1.0)
        else:
            w = np.ones(len(d))
        for c in pcols:
            d[c] = d[c] * w
        g = d.groupby(["Subject_id", "Study_id"], as_index=False)[pcols].sum()
        wsum = pd.Series(w).groupby([df["Subject_id"].values, df["Study_id"].values]).sum()
        g[pcols] = g[pcols].div(wsum.values[:, None])
        if shrink < 1.0:
            pm = g.groupby("Subject_id")[pcols].transform("mean").values
            g[pcols] = shrink * g[pcols].values + (1 - shrink) * pm
        long = g.melt(id_vars=["Subject_id", "Study_id"], value_vars=pcols,
                      var_name="cls", value_name="Probability")
        long["Predict_class"] = long["cls"].str.lstrip("p").astype(int)
        long["Probability"] = long["Probability"].astype(float).round(6)
        out = long[["Subject_id", "Study_id", "Predict_class", "Probability"]].sort_values(
            ["Subject_id", "Study_id", "Predict_class"]).reset_index(drop=True)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        out.to_csv(path, index=False, encoding="utf-8")
        return out

    if fmt == "alt":
        out = pd.DataFrame({"Dicom_id": df["Dicom_id"].values})
        if pred_label:
            out["Predict_class"] = [to_pred_label(probs[i], thr) for i in range(len(out))]
        for k in range(NUM_CLASSES):
            out[f"p{k+1}"] = np.round(probs[:, k].astype(np.float64), 6)
    else:
        out = df[["Dicom_id", "Subject_id", "Study_id"]].copy().reset_index(drop=True)
        for k in range(NUM_CLASSES):
            out[f"p{k+1}"] = np.round(probs[:, k].astype(np.float64), 6)
        if pred_label:
            out["Pred_Label"] = [to_pred_label(probs[i], thr) for i in range(len(out))]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    out.to_csv(path, index=False)
    return out


def add_extra_channel(t, name: str):
    """按训练时的配置追加额外输入通道（t 为 (1,H,W) 单通道张量）。"""
    if name != "asym":
        return t
    if t.dim() == 2:
        t = t.unsqueeze(0)
    a = (t - torch.flip(t, dims=[-1])).abs()
    a = a / a.std().clamp_min(1e-3)
    return torch.cat([t, a], dim=0)
