"""Deployment specification shared by predict.py and cost.py.

A single `model/deploy.json` lists the ensemble members and the inference
precision; both entry points load it through this module, so the timing script
and the submission use exactly the same model loading and inference code
(competition rule 7).
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .cache import image_path, load_square, load_square_views, read_split
from .config import NUM_CLASSES
from .infer import PathDataset, forward_tta, load_checkpoint, predict_probs, predict_one, preprocess

DEFAULT_DEPLOY = os.path.join("model", "deploy.json")


def apply_pooling_overrides(model: nn.Module, member: dict) -> None:
    """Optional inference-only LSE settings; absent keys preserve checkpoint behavior."""
    keys = ("lse_alpha", "lse_tau")
    if not any(key in member for key in keys):
        return
    if (str(member.get("kind", "our")) != "our"
            or getattr(model, "lse_alpha", 0) <= 0
            or getattr(model, "n_view", 0)
            or getattr(model, "query_head", None) is not None
            or getattr(model, "region_head", None) is not None):
        raise ValueError("pooling overrides require an existing LSE-only model")
    values = {}
    for key in keys:
        value = member.get(key, getattr(model, key))
        if isinstance(value, bool):
            raise ValueError(f"{key} must be a finite number")
        value = float(value)
        if not math.isfinite(value) or value <= 0 or (key == "lse_alpha" and value > 1):
            raise ValueError(f"invalid {key}: {value}")
        values[key] = value
    for key, value in values.items():
        setattr(model, key, value)


def resolve_root(deploy_path: str) -> str:
    """Paths inside deploy.json are relative to the deployment root."""
    d = os.path.dirname(os.path.abspath(deploy_path))
    # model/deploy.json -> package root; fall back to the directory itself
    parent = os.path.dirname(d)
    return parent if os.path.basename(d) == "model" else d


class DonorModel(nn.Module):
    """Wrapper for a model trained outside this repo (only inference is needed).

    The donor owns its architecture and weights; this wrapper only makes its output look
    like ours -- logits of shape (B, 10) with class 10 last -- so the existing TTA and
    weighting code works unchanged. Input is 3-channel ImageNet-normalised, produced by
    preprocess(..., norm="imagenet3") from the SAME decoded cache image our members use,
    which is why adding a donor costs compute but no extra JPEG decode.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.n_view = 0

    def forward(self, x, view=None, return_aux: bool = False):
        l10, l9 = self.model(x)
        if l10.dim() == 1:                          # donor squeezes the class-10 head
            l10 = l10.unsqueeze(1)
        out = torch.cat([l9, l10], dim=1)          # classes 1..9 then 10
        if return_aux:
            return out, None
        return out


def _autocast_state():
    """兼容 torch 2.4–2.9 的 autocast 查询接口。

    torch 较新版本中 `is_autocast_enabled()` 需要 device_type 参数，较早版本不接受；
    `get_autocast_dtype` 从 2.4 才提供。云上镜像（torch 2.5.1）与本地（2.9）都要能跑。
    """
    try:
        enabled = bool(torch.is_autocast_enabled("cuda"))
    except TypeError:
        enabled = bool(torch.is_autocast_enabled())
    try:
        dtype = torch.get_autocast_dtype("cuda")
    except (AttributeError, TypeError):
        dtype = torch.bfloat16 if enabled else None
    return enabled, dtype


class GraphModel(nn.Module):
    """Replay a fixed-shape forward through a CUDA graph.

    At batch 1 this model is kernel-launch bound: 36 ms of forward for a network that
    needs a fraction of that in FLOPs. Capturing the graph once and replaying it costs
    ~5 ms and returns bitwise-identical logits (verified max |diff| = 0), so determinism
    is preserved. One graph is captured per input shape, and any capture failure falls
    back to the eager path instead of breaking inference.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.n_view = 0
        self._graphs: dict = {}

    def _capture(self, x: torch.Tensor):
        static_in = x.detach().clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        amp_enabled, amp_dtype = _autocast_state()
        # Cast weights inside capture; do not retain external autocast-cache addresses.
        # 与交付包（11-13 冻结版 code/cxr/deploy.py）保持一致：缺这段 autocast 包裹会让
        # bf16 数值路径偏移（实测 valid 宏 0.825355 vs 包内 0.825195）。
        with torch.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype, cache_enabled=False):
            with torch.cuda.stream(side), torch.no_grad():
                for _ in range(4):
                    self.model(static_in)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(graph):
                static_out = self.model(static_in)
        return static_in, static_out, graph

    def forward(self, x, view=None, return_aux: bool = False):
        if not x.is_cuda:
            return self.model(x)
        key = tuple(x.shape)
        if key not in self._graphs:
            try:
                self._graphs[key] = self._capture(x)
            except Exception as exc:                      # capture is best-effort
                print(f"[graph] capture failed for {key}: {exc}; using eager path",
                      flush=True)
                self._graphs[key] = None
        entry = self._graphs[key]
        if entry is None:
            return self.model(x)
        static_in, static_out, graph = entry
        static_in.copy_(x, non_blocking=True)
        graph.replay()
        return static_out


def load_donor_v21(weights_path: str, device: str = "cpu") -> DonorModel:
    """Load the donor's reference implementation (architecture + preprocessing).

    This used to read the reference from the repo path `其他/v21_prep/`, which does not
    exist inside the delivered package, so reproduction there failed with a bare
    FileNotFoundError. The reference is now shipped inside the package
    (`code/cxr/donor_ref/`) and several locations are searched before giving up, so both
    the repo layout and the standalone package work.
    """
    import importlib.util

    here = os.path.dirname(os.path.abspath(__file__))            # .../code/cxr
    code_root = os.path.dirname(here)                            # .../code
    top = os.path.dirname(code_root)                             # package / repo root
    candidates = []
    if os.environ.get("CXR_DONOR_REF"):
        candidates.append(os.environ["CXR_DONOR_REF"])
    candidates += [
        os.path.join(here, "donor_ref", "preprocess_predict.py"),        # shipped copy
        os.path.join(code_root, "donor_ref", "preprocess_predict.py"),
        os.path.join(top, "其他", "v21_prep", "preprocess_predict.py"),   # repo layout
    ]
    ref = next((p for p in candidates if os.path.exists(p)), None)
    if ref is None:
        raise FileNotFoundError(
            "donor reference implementation preprocess_predict.py not found; tried: "
            + "; ".join(candidates))
    spec = importlib.util.spec_from_file_location("v21_ref", ref)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    inner = mod.load_v21(weights_path, device=device)
    return GraphModel(DonorModel(inner)).eval()


class EnsemblePredictor:
    def __init__(self, deploy_path: str, device: str | None = None):
        with open(deploy_path, encoding="utf-8") as f:
            self.spec = json.load(f)
        self.root = resolve_root(deploy_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.amp = self.spec.get("amp", "bf16")
        self.fusion = self.spec.get("fusion", "arithmetic")
        if self.fusion not in ("arithmetic", "half_power"):
            raise ValueError(f"Unsupported ensemble fusion: {self.fusion}")
        cpu_threads = self.spec.get("cpu_threads")
        if cpu_threads is not None:
            if isinstance(cpu_threads, bool) or not isinstance(cpu_threads, int) or cpu_threads < 1:
                raise ValueError("cpu_threads must be a positive integer")
            torch.set_num_threads(cpu_threads)
        if self.device == "cuda":
            # benchmark=False: cuDNN uses a deterministic heuristic instead of
            # autotuning, so two processes on the same input pick the same
            # algorithm. Benchmark=True made predictions differ by ~1e-3 between
            # runs, which is enough to flip a thresholded label (rule 7 demands
            # 0 inconsistent samples).
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        self.models = []
        for m in self.spec["members"]:
            resize_mode = m.get("resize_mode", "tensor")
            if resize_mode not in ("tensor", "pil_direct"):
                raise ValueError(f"unsupported resize_mode: {resize_mode}")
            ck = m["ckpt"]
            path = ck if os.path.isabs(ck) else os.path.join(self.root, ck)
            if str(m.get("kind", "our")) == "donor_v21":
                # externally trained member: same inference contract, own architecture
                model, meta = load_donor_v21(path, self.device), {}
            else:
                model, meta = load_checkpoint(path, self.device)
            apply_pooling_overrides(model, m)
            model.eval()
            self.models.append({
                "model": model,
                "img_size": int(m.get("img_size", meta.get("img_size", 384))),
                "cache_size": int(m.get("cache_size", 448)),
                "norm": m.get("norm", "per_image"),
                "tta": bool(m.get("tta", True)),
                "weight": float(m.get("weight", 1.0)),
                "kind": str(m.get("kind", "our")),
                "name": m.get("name", os.path.basename(path)),
                "resize_mode": resize_mode,
            })
        w = np.array([m["weight"] for m in self.models], dtype=np.float64)
        if self.fusion == "half_power" and (
                not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0):
            raise ValueError("half_power fusion requires finite nonnegative weights with positive sum")
        self.weights = w / w.sum()
        # Dicom_id -> acquisition view id, for models that take the view as an input
        self.view_of = {}
        if any(getattr(m["model"], "n_view", 0) > 0 for m in self.models):
            vmap = {"AP": 0, "PA": 1, "LATERAL": 2, "LL": 3}
            for split in ("test", "valid", "train"):
                try:
                    df = read_split(split)
                except Exception:
                    continue
                self.view_of.update({
                    d: vmap.get(str(v).strip(), 4)
                    for d, v in zip(df.Dicom_id, df.ViewPosition.fillna("NA"))})

    # ------------------------------------------------------------------ #
    def predict_frame(self, df: pd.DataFrame, split: str, batch_size: int = 16) -> np.ndarray:
        """Probabilities for a split.

        batch_size == 1 goes through predict_path(), i.e. literally the same
        call as cost.py, which makes the submission bit-identical to the timing
        script (rule 7 requires zero inconsistent samples).
        """
        if batch_size <= 1:
            out = np.zeros((len(df), NUM_CLASSES), dtype=np.float64)
            for i, r in enumerate(df.itertuples()):
                p, _ = self.predict_path(image_path(split, r.Subject_id, r.Study_id, r.Dicom_id))
                out[i] = p
            return out
        acc = np.zeros((len(df), NUM_CLASSES), dtype=np.float64)
        acc_sq = np.zeros_like(acc) if self.fusion == "half_power" else None
        wsum = 0.0
        for m in self.models:
            ds = PathDataset(df, split, img_size=m["img_size"],
                             cache_size=m["cache_size"], norm=m["norm"],
                             resize_mode=m["resize_mode"])
            p = predict_probs(m["model"], ds, device=self.device,
                              batch_size=batch_size, tta=m["tta"], amp=self.amp)
            acc += m["weight"] * p
            if acc_sq is not None:
                acc_sq += m["weight"] * np.square(p.astype(np.float64))
            wsum += m["weight"]
        return self._finish_fusion(acc, acc_sq, wsum)

    def _finish_fusion(self, acc, acc_sq, wsum):
        mean = acc / max(wsum, 1e-9)
        if self.fusion == "arithmetic":
            return mean
        return .5 * mean + .5 * np.sqrt(acc_sq / max(wsum, 1e-9))

    def predict_path(self, path: str):
        """Single-image inference (batch=1) returning (probs, elapsed_ms).

        The JPEG is decoded once and the preprocessed tensor is shared between
        members that use the same input geometry.
        """
        t0 = time.perf_counter()
        acc = np.zeros(NUM_CLASSES, dtype=np.float64)
        acc_sq = np.zeros_like(acc) if self.fusion == "half_power" else None
        wsum = 0.0
        raw = {}
        prep = {}
        direct = {}
        direct_sizes = {}
        for m in self.models:
            if m["resize_mode"] == "pil_direct":
                direct_sizes.setdefault(m["cache_size"], set()).update(
                    (m["cache_size"], m["img_size"]))
        for cache_size, sizes in direct_sizes.items():
            decoded = load_square_views(path, cache_size, sizes)
            raw[cache_size] = decoded[cache_size]
            direct.update({(cache_size, size): value for size, value in decoded.items()})
        dicom = os.path.splitext(os.path.basename(path))[0]
        view = self.view_of.get(dicom, 4)
        for m in self.models:
            if m["cache_size"] not in raw:
                raw[m["cache_size"]] = load_square(path, m["cache_size"])
            key = (m["cache_size"], m["img_size"], m["norm"], m["resize_mode"])
            if key not in prep:
                image = (direct[(m["cache_size"], m["img_size"])]
                         if m["resize_mode"] == "pil_direct" else raw[m["cache_size"]])
                prep[key] = preprocess(image, m["img_size"],
                                       m["norm"]).unsqueeze(0).to(self.device)
            p = forward_tta(m["model"], prep[key], tta=m["tta"], amp=self.amp,
                            view=view if getattr(m["model"], "n_view", 0) > 0 else None)
            acc += m["weight"] * p
            if acc_sq is not None:
                acc_sq += m["weight"] * np.square(p.astype(np.float64))
            wsum += m["weight"]
        return self._finish_fusion(acc, acc_sq, wsum), (time.perf_counter() - t0) * 1000.0


def thresholds_for(deploy_path: str) -> dict:
    root = resolve_root(deploy_path)
    p = os.path.join(os.path.dirname(deploy_path), "thresholds.json")
    if not os.path.exists(p):
        p = os.path.join(root, "reports", "valid_thresholds.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}
