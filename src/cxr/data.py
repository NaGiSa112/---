"""Torch dataset + augmentation for the chest X-ray task."""
from __future__ import annotations

import json
import math
import os
import random

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from .cache import (build_cache, image_path, label_matrix, load_square,  # noqa: F401
                    open_cache, parse_label_string, read_split)
from .config import CACHE, NUM_CLASSES


class CXRDataset(Dataset):
    """Reads the resized memmap cache and applies on-the-fly augmentation."""

    def __init__(self, split: str, cfg, train: bool, cache_dir: str = CACHE,
                 df=None, size: int | None = None):
        self.cfg = cfg
        self.train = train
        self.size = size or cfg.cache_size
        self.img_size = cfg.img_size
        self.df = read_split(split) if df is None else df.reset_index(drop=True)
        # allow a preprocessing variant cache (CLAHE / thoracic crop) without touching call sites
        if cache_dir is None:
            cache_dir = getattr(cfg, "cache_dir", "") or CACHE
        elif cache_dir == CACHE and getattr(cfg, "cache_dir", ""):
            cache_dir = cfg.cache_dir
        # A JPEG-backed cache exists so the training data can be shipped to a remote
        # machine over a slow link: the memmap is 12.96 GB at 640 px, while the same
        # pixels as quality-92 JPEG are ~1.2 GB at 512 px. Decoding costs 2-3 ms per
        # sample, which hides behind GPU compute as soon as workers > 0.
        self.jpg_dir = None
        if getattr(cfg, "jpg_cache", False):
            jdir = os.path.join(os.path.dirname(cache_dir), "cache_jpg", f"{split}_{self.size}")
            if not os.path.isdir(jdir):
                raise FileNotFoundError(
                    f"jpg cache not found: {jdir} (run tools/build_jpg_cache.py, "
                    f"or drop --jpg-cache)")
            with open(os.path.join(jdir, "meta.json"), encoding="utf-8") as f:
                cids = json.load(f)["dicom_ids"]
            self.jpg_dir = jdir
            self.cache = None
        else:
            self.cache, meta = open_cache(split, self.size, cache_dir)
            cids = meta["dicom_ids"]
        if cids != self.df["Dicom_id"].tolist():
            pos = {c: i for i, c in enumerate(cids)}
            self.order = np.array([pos[c] for c in self.df["Dicom_id"]], dtype=np.int64)
        else:
            self.order = None
        # Views of one study share their labels, so two views of the same study must
        # predict the same thing. Building the index here lets __getitem__ fetch a
        # sibling view for the consistency term (MVC-Net-style, see data.py docstring).
        self.view_groups = {}
        if self.train and float(getattr(cfg, "view_consistency", 0.0) or 0.0) > 0:
            for idx, sid in enumerate(self.df["Study_id"].tolist()):
                self.view_groups.setdefault(sid, []).append(idx)
            self.view_of_row = self.df["Study_id"].tolist()
        self.y = label_matrix(self.df) if "Predict_class" in self.df.columns else None
        # Knowledge distillation via blended targets. BCE is linear in its target:
        #   BCE(z, (1-w)y_hard + w*y_soft) == (1-w)BCE(z,y_hard) + w*BCE(z,y_soft)
        # so mixing the label matrix is exactly the weighted soft/hard objective and
        # needs no change in the training loop.
        self.distill_w = float(getattr(cfg, "distill_weight", 0.0) or 0.0)
        distill_csv = getattr(cfg, "distill_csv", "") or ""
        # Only the TRAINING set gets blended targets. The validation set keeps hard
        # labels, otherwise the reported val AUC would measure agreement with the
        # teacher instead of accuracy against the real labels.
        if self.train and self.y is not None and self.distill_w > 0 and distill_csv:
            soft = pd.read_csv(distill_csv)
            soft = soft.set_index("Dicom_id")
            missing = [c for c in self.df.Dicom_id if c not in soft.index]
            if missing:
                raise SystemExit(f"[distill] {len(missing)} images missing soft labels, "
                                 f"e.g. {missing[:3]}")
            s = soft.loc[self.df.Dicom_id, [f"p{k}" for k in range(1, 11)]].to_numpy(np.float32)
            self.y = (1.0 - self.distill_w) * self.y.astype(np.float32) + self.distill_w * s
            print(f"[distill] blended targets: w={self.distill_w} "
                  f"from {os.path.basename(distill_csv)} ({len(s)} images)")

        # Patient-level prior: a patient's studies carry correlated labels, and the test
        # split is dominated by chronic patients (mean 16.3 studies per patient vs 1.67
        # in training). Blending each image target slightly toward its own patient mean
        # teaches that structure at the target level, with no inference cost and no model
        # change -- the same linearity argument as distillation above.
        # 类 4 与类 8 的目标互混（y4' = (1-w)y4 + w·y8，y8' 同理）。
        # 依据：P(8|4)=0.529，两类在影像上连续、标注边界模糊。注意"合并两类"已被实测否定
        # （合并预测后两类都变差），这里保留两个独立头，只放宽目标。
        self.blend48 = float(getattr(cfg, "label_blend_48", 0.0) or 0.0)
        if self.train and self.y is not None and self.blend48 > 0:
            y = self.y.astype(np.float32).copy()
            c4, c8 = 3, 7
            y4, y8 = y[:, c4].copy(), y[:, c8].copy()
            y[:, c4] = (1.0 - self.blend48) * y4 + self.blend48 * y8
            y[:, c8] = (1.0 - self.blend48) * y8 + self.blend48 * y4
            self.y = y
            print(f"[blend48] class4 <-> class8 target blend w={self.blend48} "
                  f"(mean y4 {y4.mean():.3f}->{y[:, c4].mean():.3f}, "
                  f"y8 {y8.mean():.3f}->{y[:, c8].mean():.3f})")

        self.patient_w = float(getattr(cfg, "patient_smooth", 0.0) or 0.0)
        if self.train and self.y is not None and self.patient_w > 0:
            if "Subject_id" not in self.df.columns:
                raise SystemExit("[patient] Subject_id missing")
            pm = pd.DataFrame(self.y).groupby(self.df["Subject_id"].values).transform("mean")
            pm = pm.to_numpy(np.float32)
            # 允许只改某一个类的平滑强度：类 1 的正样本阳性率最低、患者内阴性检查最多，
            # 全类统一的 0.3 会把它的阳性目标稀释（实测均值 0.878，最低 0.714）。
            w_vec = np.full(10, self.patient_w, np.float32)
            c1_w = float(getattr(cfg, "patient_smooth_c1", -1.0))
            if c1_w >= 0:
                w_vec[0] = c1_w
                print(f"[patient] 类1 平滑系数单独设为 {c1_w}（其余 {self.patient_w}）")
            self.y = (1.0 - w_vec) * self.y.astype(np.float32) + w_vec * pm
            n_pat = self.df["Subject_id"].nunique()
            print(f"[patient] targets blended toward the patient mean: w={w_vec.tolist()} "
                  f"({n_pat} patients, {len(self.y)} images)")
        self.epoch = 0
        # optional targets: the acquisition view, used either as an auxiliary label
        # (aux_view_weight) or as an input feature (use_view_input)
        self.aux = None
        self.view_ids = None
        if ("ViewPosition" in self.df.columns
                and (getattr(cfg, "aux_view_weight", 0.0) > 0
                     or getattr(cfg, "use_view_input", False))):
            vmap = {"AP": 0, "PA": 1, "LATERAL": 2, "LL": 3}
            self.view_ids = np.array([vmap.get(str(v).strip(), 4)
                                      for v in self.df["ViewPosition"]], dtype=np.int64)
            if getattr(cfg, "aux_view_weight", 0.0) > 0:
                self.aux = self.view_ids
        # ---- 十类别独立增强：辅助视图策略 ----
        # 抽样与标签无关（先抽图、再均匀抽类别 k），避免"阳性弱增强/阴性强增强"的捷径。
        self.aux_branch = bool(getattr(cfg, "aux_branch", False))
        self.aux_pol = None
        if self.aux_branch:
            self.aux_pol = _load_aux_policies(getattr(cfg, "aux_policy", "perclass"),
                                              getattr(cfg, "aux_policy_json", ""))

    def set_epoch(self, epoch: int):
        """Vary the augmentation stream per epoch even when num_workers == 0."""
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.df)

    def _augment(self, img: np.ndarray, rs: random.Random) -> torch.Tensor:
        cfg = self.cfg
        h, w = img.shape
        out = self.img_size
        if self.train:
            # no_crop=True 保留完整画幅：不裁外圈、不改宽高比。
            # 动机（见 04_新一轮/04_文档/十类别独立数据增强方案）：类 1/6 的判据是心胸比与纵隔宽度，
            # 类 2/9 依赖肺尖与肋膈角等外围结构，裁剪等于直接切掉证据。
            if not getattr(cfg, "no_crop", False):
                for _ in range(10):
                    scale = rs.uniform(cfg.scale_min, 1.0)
                    ratio = rs.uniform(float(getattr(cfg, "ratio_min", 0.85)),
                                       float(getattr(cfg, "ratio_max", 1.18)))
                    area = h * w * scale
                    ch = int(round(math.sqrt(area / ratio)))
                    cw = int(round(math.sqrt(area * ratio)))
                    if ch <= h and cw <= w:
                        top = rs.randint(0, h - ch)
                        left = rs.randint(0, w - cw)
                        img = img[top:top + ch, left:left + cw]
                        break
            if rs.random() < cfg.hflip:
                img = img[:, ::-1]
        t = torch.from_numpy(np.ascontiguousarray(img)).float()[None, None]
        t = torch.nn.functional.interpolate(t, size=(out, out), mode="bilinear",
                                            align_corners=False, antialias=True)[0]
        # t is (1, out, out) from here on
        if self.train:
            if cfg.rot_deg > 0 and rs.random() < 0.35:
                t = _rotate(t, rs.uniform(-cfg.rot_deg, cfg.rot_deg))
            if cfg.rbc_prob > 0 and rs.random() < cfg.rbc_prob:
                m = t.mean()
                t = (t - m) * (1.0 + rs.uniform(-cfg.contrast, cfg.contrast)) \
                    + m + rs.uniform(-cfg.brightness, cfg.brightness) * 255.0
        return t

    def _aux_view(self, img: np.ndarray, k: int, view: str, rs: random.Random) -> np.ndarray:
        """按类别 k 的几何策略，从**原始未增强**缓存图生成一个辅助视图（uint8, out×out）。

        与主分支的区别（遵循方案文档的硬约束）：
          * 不裁剪、不改宽高比、不擦除、不 mixup —— 保留完整源画幅
          * 只用等比缩放 + 小角度旋转 + 受限平移，且**一次重采样**完成
          * 补边用统一暗背景 0，不反射/不复制解剖结构
          * 平移上限受"旋转后剩余补边余量"钳制，保证源图四角不出界
        体位回退：只有 AP/PA 用各类上限；LATERAL/LL/未知一律用统一保守参数。
        水平翻转与主视图同一规则（p=cfg.hflip），B/C 两臂共用，避免多一个未控变量。
        """
        out = int(self.img_size)
        h, w = img.shape
        if getattr(self.cfg, "hflip", 0.0) > 0 and rs.random() < float(self.cfg.hflip):
            img = np.ascontiguousarray(img[:, ::-1])
        pol = self.aux_pol or {}
        base = pol.get("uniform", {"rot": 2.0, "trans": 0.01, "r": [0.99, 1.0]})
        g = pol.get("classes", {}).get(k, base)
        if view not in ("AP", "PA"):
            g = pol.get("lateral", base)

        theta = math.radians(rs.uniform(-float(g["rot"]), float(g["rot"])))
        r = rs.uniform(float(g["r"][0]), float(g["r"][1]))
        trans_max = float(g["trans"])

        c, s = math.cos(theta), math.sin(theta)
        # 旋转后源图的轴对齐包围盒
        bw = abs(w * c) + abs(h * s)
        bh = abs(w * s) + abs(h * c)
        s_fit = min(out / bw, out / bh)          # 等比缩放，保证四角全进画布
        scale = s_fit * r
        if scale <= 0:
            scale = s_fit
        # 剩余补边余量（输出像素）——平移不得超过它，否则源角会出界
        margin_x = max(0.0, (out - bw * scale) / 2.0)
        margin_y = max(0.0, (out - bh * scale) / 2.0)
        tx = rs.uniform(-1.0, 1.0) * min(trans_max * out, margin_x)
        ty = rs.uniform(-1.0, 1.0) * min(trans_max * out, margin_y)

        cx_in, cy_in = w / 2.0, h / 2.0
        cx_out, cy_out = out / 2.0 - tx, out / 2.0 - ty
        inv = 1.0 / scale
        a = c * inv
        b = s * inv
        d = -s * inv
        e = c * inv
        # PIL AFFINE：(x_in, y_in) = (a u + b v + cc, d u + e v + ff)
        cc = cx_in - a * cx_out - b * cy_out
        ff = cy_in - d * cx_out - e * cy_out

        from PIL import Image as _Image
        src = _Image.fromarray(np.ascontiguousarray(img), mode="L")
        res = src.transform((out, out), _Image.AFFINE, (a, b, cc, d, e, ff),
                            resample=_Image.BICUBIC, fillcolor=0)
        return np.asarray(res, dtype=np.uint8)

    def __getitem__(self, i):
        j = i if self.order is None else int(self.order[i])
        if self.jpg_dir is not None:
            with Image.open(os.path.join(self.jpg_dir, f"{j:06d}.jpg")) as im:
                img = np.asarray(im.convert("L"), dtype=np.uint8)
        else:
            img = np.asarray(self.cache[j], dtype=np.uint8)
        seed = (torch.initial_seed() + self.epoch * 1_000_003 + i * 7919) % (2 ** 31)
        rs = random.Random(seed)
        x = self._augment(img, rs)          # (1, out, out)
        if self.cfg.norm == "per_image":
            x = (x - x.mean()) / x.std().clamp_min(1e-3)
        else:
            x = (x - 127.5) / 64.0
        if self.train and self.cfg.erase_prob > 0 and rs.random() < self.cfg.erase_prob:
            x = _random_erase(x, rs)
        # 可选的额外输入通道。asym = |I − fliplr(I)|：单侧改变（如单侧容积缩小）在这张图上
        # 是左右对称性破坏，而灰度图本身把它混在整体密度里。第一层卷积的新通道零初始化，
        # 因此模型第 0 步与单通道版本完全等价，只在该特征有用时才会偏离。
        if getattr(self.cfg, "extra_channel", "") == "asym":
            a = (x - torch.flip(x, dims=[2])).abs()
            a = a / a.std().clamp_min(1e-3)
            x = torch.cat([x, a], dim=0)
        y = torch.zeros(NUM_CLASSES) if self.y is None else torch.from_numpy(self.y[i])
        # ---- 辅助视图分支：与主分支完全独立的一条返回路径 ----
        # 每张图以 aux_frac 的概率成为辅助图（不看标签），再均匀抽一个目标类别 k，
        # 只用该类别的几何策略生成第二视图；返回 6 元组，由 collate 单独路由。
        if self.aux_branch and self.train:
            if rs.random() < float(getattr(self.cfg, "aux_frac", 0.25)):
                k = rs.randrange(NUM_CLASSES)
                vp = ""
                if "ViewPosition" in self.df.columns:
                    vp = str(self.df["ViewPosition"].iloc[i]).strip()
                aux_np = self._aux_view(img, k, vp, rs)
                aux_x = torch.from_numpy(aux_np)[None].float()
                if self.cfg.norm == "per_image":
                    aux_x = (aux_x - aux_x.mean()) / aux_x.std().clamp_min(1e-3)
                else:
                    aux_x = (aux_x - 127.5) / 64.0
                vid = int(self.view_ids[i]) if self.view_ids is not None else -1
                return (x, y, torch.tensor(vid), aux_x, torch.tensor(k),
                        torch.tensor(1.0))
            vid = int(self.view_ids[i]) if self.view_ids is not None else -1
            return (x, y, torch.tensor(vid), torch.zeros(1, x.shape[-2], x.shape[-1]),
                    torch.tensor(0), torch.tensor(0.0))
        if self.aux is not None:
            return x, y, torch.tensor(int(self.aux[i]))
        if self.view_groups:
            x2 = self._second_view(i, y)
            if self.view_ids is not None:
                return x, y, torch.tensor(-1), torch.tensor(int(self.view_ids[i])), x2
            return x, y, torch.tensor(-1), torch.tensor(-1), x2
        if self.view_ids is not None:
            # view carried as a 4th element so collate can route it to the model
            return x, y, torch.tensor(-1), torch.tensor(int(self.view_ids[i]))
        return x, y


    def _second_view(self, i, y):
        """Another image of the same study, independently augmented.

        Falls back to the same image when the study has a single view, which makes the
        consistency term exactly zero for that sample instead of injecting noise.
        """
        sid = self.view_of_row[i]
        sibs = self.view_groups.get(sid, [])
        j = i
        if len(sibs) > 1:
            j = sibs[(sibs.index(i) + 1) % len(sibs)]
            if j == i:
                j = sibs[0]
        idx = j if self.order is None else int(self.order[j])
        img = np.asarray(self.cache[idx], dtype=np.uint8) if self.jpg_dir is None else None
        if img is None:
            with Image.open(os.path.join(self.jpg_dir, f"{idx:06d}.jpg")) as im:
                img = np.asarray(im.convert("L"), dtype=np.uint8)
        seed = (torch.initial_seed() + self.epoch * 7_777_777 + i * 31 + 1) % (2 ** 31)
        rs = random.Random(seed)
        x2 = self._augment(img, rs)
        if self.cfg.norm == "per_image":
            x2 = (x2 - x2.mean()) / x2.std().clamp_min(1e-3)
        else:
            x2 = (x2 - 127.5) / 64.0
        if self.cfg.erase_prob > 0 and rs.random() < self.cfg.erase_prob:
            x2 = _random_erase(x2, rs)
        return x2


def _rotate(t: torch.Tensor, angle: float) -> torch.Tensor:
    _, h, w = t.shape
    theta = math.radians(angle)
    cos, sin = math.cos(theta), math.sin(theta)
    grid = torch.tensor([[[cos, -sin, 0.0], [sin, cos, 0.0]]], dtype=t.dtype)
    grid = torch.nn.functional.affine_grid(grid, (1, 1, h, w), align_corners=False)
    out = torch.nn.functional.grid_sample(t.unsqueeze(0), grid, align_corners=False,
                                          mode="bilinear", padding_mode="border")
    return out[0]


def _random_erase(x: torch.Tensor, rs: random.Random, p: float = 0.15) -> torch.Tensor:
    _, h, w = x.shape
    area = h * w * rs.uniform(0.02, p)
    ratio = rs.uniform(0.4, 2.5)
    eh = int(round(math.sqrt(area * ratio)))
    ew = int(round(math.sqrt(area / ratio)))
    if eh >= h or ew >= w:
        return x
    top = rs.randint(0, h - eh)
    left = rs.randint(0, w - ew)
    x[:, top:top + eh, left:left + ew] = x.mean()
    return x


def collate(batch):
    xs = torch.stack([b[0] for b in batch])
    ys = torch.stack([b[1] for b in batch])
    if len(batch[0]) == 6:                      # 十类别增强的辅助视图分支
        # (x, y, view_id, aux_x, aux_k, aux_mask)；未抽中的样本 aux_x 为零图、mask=0
        views = torch.stack([b[2] for b in batch])
        aux_x = torch.stack([b[3] for b in batch])
        aux_k = torch.stack([b[4] for b in batch])
        aux_m = torch.stack([b[5] for b in batch])
        return xs, ys, views, aux_x, aux_k, aux_m
    if len(batch[0]) > 4:                       # second view for consistency training
        aux = torch.stack([b[2] for b in batch])
        views = torch.stack([b[3] for b in batch])
        x2 = torch.stack([b[4] for b in batch])
        return xs, ys, aux, views, x2
    if len(batch[0]) > 2:
        aux = torch.stack([b[2] for b in batch])
        if len(batch[0]) > 3:
            views = torch.stack([b[3] for b in batch])
            # aux == -1 means "no aux target, view is an input feature"
            return xs, ys, aux, views
        return xs, ys, aux
    return xs, ys


def _load_aux_policies(kind: str, json_path: str = "") -> dict:
    """读取十类别增强策略。

    kind="uniform"    → B 组对照：统一温和策略（±2°、1%、r 0.99–1.0）
    kind="perclass"   → C 组候选：读 JSON 的十类策略
    体位回退两种都共用（方案要求）。
    """
    uniform = {"rot": 2.0, "trans": 0.01, "r": [0.99, 1.0]}
    lateral = {"rot": 0.5, "trans": 0.005, "r": [0.995, 1.0]}
    if kind == "uniform":
        return {"classes": {}, "uniform": uniform, "lateral": lateral}
    if not json_path:
        json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "04_新一轮", "04_文档",
            "十类别增强参数草案_2026-11-25.json")
    if not os.path.exists(json_path):
        print(f"[aux] 策略文件缺失，退回统一策略：{json_path}")
        return {"classes": {}, "uniform": uniform, "lateral": lateral}
    with open(json_path, encoding="utf-8") as f:
        d = json.load(f)
    classes = {}
    for p in d.get("class_policies", []):
        g = p.get("geometry", {})
        classes[int(p["id"]) - 1] = {
            "rot": float(g.get("rotation_max_deg", 2.0)),
            "trans": float(g.get("translation_max_fraction", 0.01)),
            "r": [float(x) for x in g.get("relative_fit_scale_range", [0.99, 1.0])],
        }
    fb = d.get("view_fallback", {}).get("lateral_ll_or_unknown", {})
    if fb:
        lateral = {"rot": float(fb.get("rotation_max_deg", 0.5)),
                   "trans": float(fb.get("translation_max_fraction", 0.005)),
                   "r": [float(x) for x in fb.get("relative_fit_scale_range", [0.995, 1.0])]}
    print(f"[aux] 已载入十类策略 {len(classes)} 类；体位回退 rot±{lateral['rot']}° "
          f"trans {lateral['trans']:.1%}")
    return {"classes": classes, "uniform": uniform, "lateral": lateral}
