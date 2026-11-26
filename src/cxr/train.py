"""Training entry point for the chest X-ray multi-label classifier."""
from __future__ import annotations

import argparse
import json
import math
import re
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import LABELS, LABEL_NAMES, NUM_CLASSES, RUNS, TrainConfig
from .data import CXRDataset, collate, read_split, label_matrix
from .metrics import mean_auc, per_class_auc, best_f1_threshold, f1_at, patient_smooth
from .models import ModelEMA, build_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_optimizer(model, cfg: TrainConfig):
    """AdamW with optional layer-wise LR decay over the backbone stages.

    ConvNeXt fine-tuning normally scales the LR of stage i by decay**(n-1-i) so the
    stem moves least and the deepest stage most. With decay=1.0 this collapses to the
    original two-group setup.
    """
    decay = float(getattr(cfg, "layer_decay", 1.0) or 1.0)
    head, stages = [], {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("head"):
            head.append(p)
            continue
        # torchvision trunks live at backbone.<outer>.<stage>.<block>..., so the second
        # numeric component is the stage index and makes a usable depth proxy.
        nums = re.findall(r"\d+", name)
        idx = int(nums[1]) if len(nums) > 1 else -1
        stages.setdefault(idx, []).append(p)
    if decay >= 0.999 or len(stages) < 2:
        backbone = [p for g in stages.values() for p in g]
        return torch.optim.AdamW(
            [{"params": backbone, "lr": cfg.lr_backbone},
             {"params": head, "lr": cfg.lr_head}],
            weight_decay=cfg.weight_decay,
        )
    top = max(stages)
    groups = []
    for idx in sorted(stages):
        scale = decay ** (top - idx) if idx >= 0 else decay ** top   # stem gets the least
        groups.append({"params": stages[idx], "lr": cfg.lr_backbone * scale})
        print(f"[opt] stage {idx}: lr={cfg.lr_backbone * scale:.2e} ({len(stages[idx])} tensors)",
              flush=True)
    groups.append({"params": head, "lr": cfg.lr_head})
    return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)


class CosineSchedule:
    def __init__(self, opt, warmup_steps: int, total_steps: int, base_lrs):
        self.opt = opt
        self.warmup = max(warmup_steps, 1)
        self.total = max(total_steps, self.warmup + 1)
        self.base_lrs = base_lrs
        self.step_n = 0

    def step(self):
        self.step_n += 1
        if self.step_n <= self.warmup:
            f = self.step_n / self.warmup
        else:
            t = (self.step_n - self.warmup) / max(self.total - self.warmup, 1)
            f = 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))
        for g, lr in zip(self.opt.param_groups, self.base_lrs):
            g["lr"] = lr * f


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    for batch in loader:
        x, y = batch[0], batch[1]
        view = batch[3] if len(batch) > 3 else None
        x = x.to(device, non_blocking=True)
        if view is not None:
            view = view.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(x, view=view)
        ps.append(torch.sigmoid(logits.float()).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def pairwise_auc_loss(probs, y, temp=0.5, eps=1e-7):
    """Soft pairwise-ranking loss: same-class positives should outrank negatives.

    BCE calibrates probabilities; macro AUC only asks whether, within a class, a positive
    study is ranked above a negative one. For rare classes (class 1 is 3.5% positive) the
    two objectives diverge, so this term optimises the metric directly. Classes whose
    batch happens to hold only one side are skipped.
    """
    losses = []
    for k in range(y.shape[1]):
        pos_mask = y[:, k] > 0.5
        if pos_mask.sum() == 0 or (~pos_mask).sum() == 0:
            continue
        pos = probs[pos_mask, k]
        neg = probs[~pos_mask, k]
        d = (pos.unsqueeze(1) - neg.unsqueeze(0)) / temp
        losses.append(torch.nn.functional.softplus(-d).mean())
    if not losses:
        return probs.sum() * 0.0
    return torch.stack(losses).mean()


def anyfinding_target(y):
    """1 - p(class 10): class 10 is exclusive with 1..9, so this is a valid binary task."""
    return (1.0 - y[:, -1:]).clamp(0.0, 1.0)


def _mixup(x, y, alpha):
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[perm], lam * y + (1 - lam) * y[perm]


def _cutmix(x, y, alpha):
    """Rectangular CutMix: paste a random box from another sample, mix labels by area."""
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    _, _, h, w = x.shape
    rh, rw = int(h * np.sqrt(1 - lam)), int(w * np.sqrt(1 - lam))
    if rh <= 1 or rw <= 1:
        return x, y
    cy, cx = np.random.randint(h), np.random.randint(w)
    y1, y2 = max(cy - rh // 2, 0), min(cy + rh // 2, h)
    x1, x2 = max(cx - rw // 2, 0), min(cx + rw // 2, w)
    x[:, :, y1:y2, x1:x2] = x[perm][:, :, y1:y2, x1:x2]
    lam_eff = 1.0 - (y2 - y1) * (x2 - x1) / float(h * w)
    return x, lam_eff * y + (1 - lam_eff) * y[perm]


def stratified_subsample(df, n: int, seed: int = 0):
    """Take n rows keeping every label's positive rate as close as possible.

    Labels co-occur heavily and the rarest class has <3% positives, so a plain
    random subsample can lose most of them; this walks the classes from rarest
    to most common and tops up their positive examples.
    """
    if n >= len(df):
        return df
    y = label_matrix(df)
    order = np.argsort(y.sum(0))            # rarest first
    rng = np.random.default_rng(seed)
    chosen, taken = [], np.zeros(len(df), dtype=bool)
    per_label = n / len(order) / max(y.sum(0).max(), 1)
    for k in order:
        pos = np.where((y[:, k] == 1) & ~taken)[0]
        rng.shuffle(pos)
        quota = max(int(round(y[:, k].sum() * n / len(df))), 1)
        pick = pos[:quota]
        chosen.append(pick)
        taken[pick] = True
    chosen = np.concatenate(chosen)
    rest = np.where(~taken)[0]
    need = n - len(chosen)
    if need > 0:
        chosen = np.concatenate([chosen, rng.choice(rest, min(need, len(rest)), replace=False)])
    return df.iloc[np.sort(chosen)].reset_index(drop=True)


def patient_fold_split(df, fold: int, n_folds: int, seed: int = 0):
    """Patient-grouped k-fold split of train.csv (no patient leaks across folds).

    Using CV instead of the single 228-patient valid split fixes two problems at
    once: each estimate is based on ~5x more patients (the absolute AUC 95% CI on
    valid is +-0.032, which is why single-split model selection kept picking noise),
    and the held-out folds can be bagged into the final ensemble.
    """
    subj = df["Subject_id"].values
    uniq = np.unique(subj)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    groups = np.array_split(uniq, n_folds)
    hold = set(groups[fold % n_folds].tolist())
    mask = np.array([s in hold for s in subj])
    tr = df[~mask].reset_index(drop=True)
    va = df[mask].reset_index(drop=True)
    print(f"[cv] fold {fold}/{n_folds}: train {len(tr)} imgs / {len(set(tr.Subject_id))} patients, "
          f"holdout {len(va)} imgs / {len(set(va.Subject_id))} patients")
    return tr, va


def train(cfg: TrainConfig):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = cfg.tag or cfg.arch
    run_dir = cfg.out_dir or os.path.join(
        RUNS, f"{tag}_{cfg.img_size}_{time.strftime('%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2, ensure_ascii=False)
    print(f"[train] run_dir={run_dir} device={device}")

    train_df = read_split("train")
    valid_df = read_split("valid")
    if getattr(cfg, "fold_csv", ""):
        # external patient-grouped fold table (e.g. a teammate's fold_split.csv):
        # train/valid are chosen by explicit fold ids so the evaluation fold can be
        # kept completely unseen (no checkpoint selection on it).
        import pandas as _pd
        fk = _pd.read_csv(cfg.fold_csv, usecols=["Study_id", "fold"])
        fmap = dict(zip(fk.Study_id.astype(str), fk.fold.astype(int)))
        f_of = train_df.Study_id.astype(str).map(fmap)
        train_folds = [int(x) for x in cfg.train_folds.split(",") if x.strip()]
        valid_fold = int(cfg.valid_fold)
        if f_of.isna().any():
            raise SystemExit(f"[error] {int(f_of.isna().sum())} studies missing from {cfg.fold_csv}")
        tr_mask = f_of.isin(train_folds).values
        va_mask = (f_of == valid_fold).values
        valid_df = train_df[va_mask].reset_index(drop=True)
        train_df = train_df[tr_mask].reset_index(drop=True)
        print(f"[train] external fold csv: train folds {train_folds} -> {len(train_df)} images ({train_df.Study_id.nunique()} studies); "
              f"valid fold {valid_fold} -> {len(valid_df)} images ({valid_df.Study_id.nunique()} studies)")
    elif cfg.cv_folds and cfg.cv_folds > 1:
        # patient-grouped CV inside train.csv; the official valid set stays untouched
        # as an extra (never-used-for-selection) sanity check
        train_df, valid_df = patient_fold_split(train_df, cfg.cv_fold, cfg.cv_folds, cfg.seed)
        print(f"[train] CV mode: holdout is a patient fold of train.csv, not official valid")
    if cfg.view_filter:
        keep = lambda d: d[d.ViewPosition.fillna("NA").astype(str).str.upper() == cfg.view_filter.upper()].reset_index(drop=True)
        train_df = keep(train_df)
        print(f"[train] view filter {cfg.view_filter}: train -> {len(train_df)} images")
    if cfg.limit_train and cfg.limit_train < len(train_df):
        if cfg.stratify:
            train_df = stratified_subsample(train_df, cfg.limit_train, cfg.seed)
        else:
            train_df = train_df.sample(cfg.limit_train, random_state=cfg.seed).reset_index(drop=True)
        print(f"[train] stratified subsample -> {len(train_df)} images")
    if cfg.chronic_min > 0 and cfg.chronic_factor > 1.0:
        import pandas as pd
        per = train_df.groupby("Subject_id").Study_id.nunique()
        n_study = train_df.Subject_id.map(per).astype(int)
        m = (n_study >= cfg.chronic_min).values
        extra = train_df[m]
        n_extra = int(round(len(extra) * (cfg.chronic_factor - 1.0)))
        if n_extra > 0:
            pick = extra.sample(n=n_extra, replace=True, random_state=cfg.seed)
            train_df = pd.concat([train_df, pick], ignore_index=True)
            print(f"[train] chronic oversample (patients with >= {cfg.chronic_min} studies) "
                  f"x{cfg.chronic_factor}: +{n_extra} studies -> {len(train_df)} total "
                  f"({(n_study.map(lambda x: x>=cfg.chronic_min).sum()+n_extra)/len(train_df)*100:.1f}% chronic)")
    if cfg.oversample_view and cfg.oversample_factor > 1.0:
        import pandas as pd
        m = (train_df.ViewPosition.fillna("NA").astype(str).str.upper()
             == cfg.oversample_view.upper()).values
        extra = train_df[m]
        n_extra = int(round(len(extra) * (cfg.oversample_factor - 1.0)))
        if n_extra > 0:
            pick = extra.sample(n=min(n_extra, len(extra) * 3), replace=True, random_state=cfg.seed)
            train_df = pd.concat([train_df, pick], ignore_index=True)
            print(f"[train] oversample {cfg.oversample_view} x{cfg.oversample_factor}: "
                  f"+{len(pick)} images -> {len(train_df)} total")
    if cfg.merge_valid:
        # 不能把 valid 直接拼进 train_df：两个 split 的 Dicom_id 存在各自的缓存文件里，
        # 单个 CXRDataset 只能打开自己 split 的缓存（拼进来会 KeyError）。
        # 正确做法是下面用 ConcatDatasetWithLabels 把两个 dataset 连起来。
        print(f"[train] merged valid -> {len(train_df) + len(valid_df)} images "
              f"(train {len(train_df)} + valid {len(valid_df)})")
        print("[train] 注意：此模式下验证集已参与训练，验证 AUC 会偏乐观；"
              "提交时取 last.pt（固定配方跑满全程，不做 epoch 选择）")

    ds_tr = CXRDataset("train", cfg, train=True, df=train_df)
    if cfg.merge_valid:
        ds_tr = ConcatDatasetWithLabels(ds_tr, CXRDataset("valid", cfg, train=True, df=valid_df))
    if (cfg.cv_folds and cfg.cv_folds > 1) or getattr(cfg, "fold_csv", ""):
        # the holdout lives in train.csv (either our own patient fold or an external
        # fold table), so it must read the train cache -- not the official valid split
        ds_va = CXRDataset("train", cfg, train=False, df=valid_df, size=cfg.cache_size)
    else:
        ds_va = CXRDataset("valid", cfg, train=False)

    loader_kw = dict(batch_size=cfg.batch_size, num_workers=cfg.workers,
                     pin_memory=True, persistent_workers=cfg.workers > 0,
                     collate_fn=collate, drop_last=True)
    dl_tr = DataLoader(ds_tr, shuffle=True, **loader_kw)
    dl_va = DataLoader(ds_va, shuffle=False, batch_size=cfg.batch_size,
                       num_workers=cfg.workers, pin_memory=True, collate_fn=collate)

    n_aux = cfg.n_aux if cfg.aux_view_weight > 0 else 0
    n_view = cfg.n_aux if cfg.use_view_input else 0
    in_ch = 2 if getattr(cfg, "extra_channel", "") == "asym" else 1
    model = build_model(cfg.arch, pretrained=True, drop_path=cfg.drop_path,
                        n_aux=n_aux, n_view=n_view, final_norm=cfg.final_norm,
                        lse_alpha=cfg.lse_alpha, lse_tau=cfg.lse_tau,
                  head_type=getattr(cfg, "head_type", "lse"), in_ch=in_ch,
                  # timm 架构（SwinV2 等）必须按目标分辨率构造，否则默认 256 前向即崩
                  img_size=int(getattr(cfg, "img_size", 0) or 0)).to(device)
    from torchvision.ops import StochasticDepth
    actual_sd = [m.p for m in model.modules() if isinstance(m, StochasticDepth)]
    if cfg.arch.startswith("convnext") and actual_sd:
        if not math.isclose(max(actual_sd), cfg.drop_path, rel_tol=1e-9, abs_tol=1e-12):
            raise RuntimeError(
                f"ConvNeXt drop_path mismatch: requested {cfg.drop_path}, "
                f"actual maximum {max(actual_sd)}")
    print(f"[train] head={getattr(cfg, 'head_type', 'lse')} final_norm={cfg.final_norm} "
          f"lse_alpha={cfg.lse_alpha} tau={cfg.lse_tau} "
          f"actual_stochastic_depth={actual_sd}", flush=True)
    if cfg.init_ckpt:
        ck = torch.load(cfg.init_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"] if "model" in ck else ck, strict=False)
        print(f"[train] warm-started from {cfg.init_ckpt}")
    if getattr(cfg, "init_weights", ""):
        # 只替换主干：分类头保留本次运行随机初始化出来的那一份，
        # 这样与对照臂（默认 ImageNet-1K）的唯一差异就是预训练来源。
        raw = torch.load(cfg.init_weights, map_location="cpu", weights_only=False)
        sd = raw.get("model", raw)
        bb = {k: v for k, v in sd.items() if k.startswith("backbone.")}
        cur = model.state_dict()
        missing = [k for k in cur if k.startswith("backbone.") and k not in bb]
        bad = [k for k, v in bb.items()
               if k not in cur or tuple(cur[k].shape) != tuple(v.shape)]
        if missing or bad:
            raise SystemExit(f"[init] 主干权重核对失败: missing={missing[:3]} bad={bad[:3]}")
        before = {k: v.detach().cpu().clone() for k, v in cur.items()
                  if k.startswith("backbone.")}      # state_dict 是引用，先存快照
        model.load_state_dict(bb, strict=False)
        n_changed = sum(1 for k in bb if not torch.equal(bb[k].cpu(), before[k]))
        print(f"[init] backbone ← {cfg.init_weights}: {len(bb)} 个张量全部核对通过，"
              f"其中 {n_changed} 个与默认初始化不同；分类头保持本次随机初始化")
    if n_aux:
        print(f"[train] auxiliary view head enabled (n_aux={n_aux}, weight={cfg.aux_view_weight})")
    n_param = sum(p.numel() for p in model.parameters())
    print(f"[train] {cfg.arch}: {n_param/1e6:.1f}M params")

    y_tr = label_matrix(train_df)
    pos = y_tr.sum(0)
    neg = len(y_tr) - pos
    if cfg.pos_weight_power > 0:
        pw = (neg / np.maximum(pos, 1)) ** cfg.pos_weight_power
        print("[train] pos_weight", np.round(pw, 2).tolist())
    else:
        pw = np.ones(NUM_CLASSES)
    pos_weight = torch.tensor(pw, dtype=torch.float32, device=device)

    opt = make_optimizer(model, cfg)
    steps_per_epoch = len(dl_tr)
    total_steps = cfg.max_iters if cfg.max_iters > 0 else cfg.epochs * steps_per_epoch
    sched = CosineSchedule(opt, int(cfg.warmup_epochs * steps_per_epoch), total_steps,
                           [cfg.lr_backbone, cfg.lr_head])
    ema = ModelEMA(model, cfg.ema_decay) if cfg.ema_decay > 0 else None
    amp_dtype = torch.bfloat16 if cfg.amp == "bf16" else (torch.float16 if cfg.amp == "fp16" else None)
    # torch.amp.GradScaler 是 2.3+ 的写法；torch 2.1/2.2（AutoDL 的 V100 镜像）只有
    # torch.cuda.amp.GradScaler。这里两种都兼容，免得在远程为了一个 scaler 卡住。
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp_dtype == torch.float16)

    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    ce = nn.CrossEntropyLoss()

    def criterion(logits, y):
        """Dispatch on cfg.loss. ASL/focal reweight easy examples instead of classes,
        which matters here because 5 of the 10 classes sit below 0.80 AUC and macro
        AUC gives each class the same weight.

        ASL (Ridnik et al., 2021) with soft targets: the positive and negative terms
        are written so a target in [0,1] (distillation) still produces a proper value.
        """
        if getattr(cfg, "loss", "bce") == "asl":
            p = torch.sigmoid(logits)
            gp = float(getattr(cfg, "asl_gamma_pos", 0.0))
            gn = float(getattr(cfg, "asl_gamma_neg", 4.0))
            clip = float(getattr(cfg, "asl_clip", 0.05))
            p_shift = (p - clip).clamp(min=0.0) if clip > 0 else p
            lp = torch.log(p.clamp(min=1e-8))
            ln = torch.log((1.0 - p_shift).clamp(min=1e-8))
            pos = y * torch.pow((1.0 - p).clamp(min=0.0), gp) * lp
            neg = (1.0 - y) * torch.pow(p_shift, gn) * ln
            return -(pos + neg).mean()
        if getattr(cfg, "loss", "bce") == "focal":
            p = torch.sigmoid(logits)
            g = float(getattr(cfg, "asl_gamma_neg", 2.0))
            lp = torch.log(p.clamp(min=1e-8))
            ln = torch.log((1.0 - p).clamp(min=1e-8))
            return -(y * torch.pow((1.0 - p).clamp(min=0.0), g) * lp
                     + (1.0 - y) * torch.pow(p, g) * ln).mean()
        return bce(logits, y)
    best = {"epoch": -1, "auc": -1.0}
    history = []
    global_step = 0
    stop = False

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        if hasattr(ds_tr, "set_epoch"):
            ds_tr.set_epoch(epoch)
        elif hasattr(ds_tr, "ds"):
            for d in ds_tr.ds:
                d.set_epoch(epoch)
        t0 = time.time()
        run_loss = 0.0
        n_batches = 0
        for it, batch in enumerate(dl_tr, 1):
            # ---- 十类别增强：6 元组走独立路径，主路径代码完全不变 ----
            # (x, y, view_id, aux_x, aux_k, aux_mask)
            if getattr(cfg, "aux_branch", False) and len(batch) == 6:
                x = batch[0].to(device, non_blocking=True)
                y = batch[1].to(device, non_blocking=True)
                view_in = batch[2].to(device, non_blocking=True)
                aux_x = batch[3].to(device, non_blocking=True)
                aux_k = batch[4].to(device, non_blocking=True).long()
                aux_mask = batch[5].to(device, non_blocking=True).float()
                x2 = None
                v = None
                y_src = y.clone()          # mixup 之前的源标签副本（辅助项专用）
            else:
                x, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
                aux_x = aux_k = aux_mask = None
                y_src = None
                x2 = batch[4].to(device, non_blocking=True) if len(batch) > 4 else None
                v = batch[2].to(device, non_blocking=True) if len(batch) > 2 else None
                view_in = batch[3].to(device, non_blocking=True) if len(batch) > 3 else None
                if v is not None and (v < 0).all():      # aux disabled, view is an input
                    v = None
            if cfg.mixup > 0 and cfg.cutmix > 0:
                if random.random() < 0.5:
                    x, y = _mixup(x, y, cfg.mixup)
                else:
                    x, y = _cutmix(x, y, cfg.cutmix)
            elif cfg.mixup > 0:
                x, y = _mixup(x, y, cfg.mixup)
            elif cfg.cutmix > 0:
                x, y = _cutmix(x, y, cfg.cutmix)
            if cfg.label_smooth > 0:
                y = y * (1 - cfg.label_smooth) + 0.5 * cfg.label_smooth
                if y_src is not None:
                    # 两组共用同一平滑，保证辅助目标与主目标同口径
                    y_src = y_src * (1 - cfg.label_smooth) + 0.5 * cfg.label_smooth
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                if v is not None and cfg.aux_view_weight > 0:
                    logits, aux = model(x, return_aux=True, view=view_in)
                    loss = criterion(logits.float(), y) + cfg.aux_view_weight * ce(aux.float(), v)
                else:
                    logits = model(x, view=view_in)
                    loss = criterion(logits.float(), y)
                if aux_x is not None and float(getattr(cfg, "aux_weight", 0.0)) > 0 \
                        and float(aux_mask.sum()) > 0:
                    # 辅助项：每张辅助图只监督它被指定的那一类，逐张 BCE 后取均值。
                    # 注意不能把其余九类 mask 成 0 后对整个 B×10 求均值，那会稀释 10 倍。
                    # 计算量：先把命中的样本挑出来再前向。25% 抽样就该只付 25% 的算力，
                    # 否则整批 16 张都要过一遍前向，抽样等于没省。
                    hit = aux_mask > 0
                    ax = aux_x[hit]
                    ak = aux_k[hit].clamp(0, 9).view(-1, 1)
                    av = view_in[hit] if view_in is not None else None
                    logits_aux = model(ax, view=av).float()
                    sel = torch.gather(logits_aux, 1, ak).squeeze(1)
                    tgt = torch.gather(y_src[hit], 1, ak).squeeze(1)
                    l_aux = torch.nn.functional.binary_cross_entropy_with_logits(
                        sel, tgt, reduction="mean")
                    loss = loss + float(cfg.aux_weight) * l_aux
                if getattr(cfg, "rank_weight", 0.0) > 0:
                    loss = loss + float(cfg.rank_weight) * pairwise_auc_loss(
                        torch.sigmoid(logits.float()), y)
                if getattr(cfg, "any_aux_weight", 0.0) > 0 and getattr(model, "_last_any", None) is not None:
                    # plain BCE: the main criterion carries a 10-dim pos_weight that cannot
                    # broadcast against a single-logit head
                    loss = loss + float(cfg.any_aux_weight) * (
                        torch.nn.functional.binary_cross_entropy_with_logits(
                            model._last_any.float(), anyfinding_target(y)))
                if x2 is not None and getattr(cfg, "view_consistency", 0.0) > 0:
                    # Two views of one study share their labels, so their predictions
                    # should agree. MSE on probabilities is bounded and needs no
                    # temperature, which keeps this term from dominating BCE.
                    logits2 = model(x2, view=view_in)
                    loss = loss + float(cfg.view_consistency) * torch.nn.functional.mse_loss(
                        torch.sigmoid(logits.float()), torch.sigmoid(logits2.float()))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema:
                ema.update(model)
            run_loss += loss.item()
            n_batches += 1
            global_step += 1
            if it % 100 == 0:
                print(f"  e{epoch} {it}/{steps_per_epoch} loss={run_loss/n_batches:.4f} "
                      f"lr={opt.param_groups[0]['lr']:.2e} {time.time()-t0:.0f}s", flush=True)
            if cfg.max_iters > 0 and global_step >= cfg.max_iters:
                stop = True
                break

        eval_model = ema.ema if ema else model
        y_va, p_va = evaluate(eval_model, dl_va, device, amp_dtype)
        auc, aucs = mean_auc(y_va, p_va)
        p_smooth = patient_smooth(valid_df, p_va)
        auc_s, _ = mean_auc(y_va, p_smooth)
        rec = {"epoch": epoch, "loss": run_loss / max(n_batches, 1), "val_auc": auc,
               "val_auc_patient": auc_s, "per_class_auc": aucs.tolist(),
               "sec": round(time.time() - t0, 1), "global_step": global_step}
        history.append(rec)
        print(f"[e{epoch}] loss={rec['loss']:.4f} val_auc={auc:.4f} "
              f"patient_auc={auc_s:.4f} per_class={np.round(aucs,3).tolist()} "
              f"({rec['sec']}s)", flush=True)

        score = max(auc, auc_s) if cfg.merge_valid is False else auc
        if score > best["auc"]:
            best = {"epoch": epoch, "auc": score, "val_auc": auc, "val_auc_patient": auc_s}
            torch.save({"model": eval_model.state_dict(), "arch": cfg.arch,
                        "img_size": cfg.img_size, "n_aux": n_aux, "n_view": n_view, "auc": auc, "auc_patient": auc_s,
                        "epoch": epoch, "cfg": cfg.__dict__},
                       os.path.join(run_dir, "best.pt"))
        torch.save({"model": eval_model.state_dict(), "arch": cfg.arch,
                    "img_size": cfg.img_size, "n_aux": n_aux, "n_view": n_view, "epoch": epoch, "cfg": cfg.__dict__},
                   os.path.join(run_dir, "last.pt"))
        with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump({"history": history, "best": best,
                       "labels": LABELS, "label_names": LABEL_NAMES}, f, indent=2)
        if stop:
            print(f"[train] reached max_iters={cfg.max_iters} at epoch {epoch}")
            break

    print(f"[train] best={best}")
    return run_dir, best


class ConcatDatasetWithLabels(torch.utils.data.Dataset):
    def __init__(self, a, b):
        self.ds = [a, b]
        self.lens = [len(a), len(b)]
        self.n = sum(self.lens)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if i < self.lens[0]:
            return self.ds[0][i]
        return self.ds[1][i - self.lens[0]]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="convnext_tiny")
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--cache-size", type=int, default=448,
                    help="448 = 现有缓存；512/640 需要先 build_cache 重建")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--mixup", type=float, default=0.0)
    ap.add_argument("--cutmix", type=float, default=0.0)
    ap.add_argument("--label-smooth", type=float, default=0.0)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--drop-path", type=float, default=0.1)
    ap.add_argument("--final-norm", action="store_true",
                    help="retain pretrained ConvNeXt post-pooling LayerNorm (opt-in)")
    ap.add_argument("--ema", dest="ema_decay", type=float, default=0.0,
                    help="EMA decay; 0 disables (dest name must match TrainConfig.ema_decay)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--merge-valid", action="store_true")
    ap.add_argument("--init-ckpt", default="")
    ap.add_argument("--warmup-epochs", type=float, default=1.0)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-iters", type=int, default=0)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--pos-weight-power", type=float, default=0.0)
    ap.add_argument("--aux-view-weight", type=float, default=0.0)
    ap.add_argument("--erase-prob", type=float, default=0.25)
    ap.add_argument("--rbc-prob", type=float, default=0.25)
    ap.add_argument("--rot-deg", type=float, default=7.0)
    ap.add_argument("--scale-min", type=float, default=0.75)
    ap.add_argument("--hflip", type=float, default=0.5, help="水平翻转概率（默认 0.5）")
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "fp32"],
                    help="bf16 需要 Ampere+；V100(Tesla) 只支持 fp16，会自动启用 GradScaler")
    ap.add_argument("--distill-csv", default="", help="老师软标签 CSV（Dicom_id + p1..p10）")
    ap.add_argument("--distill-weight", type=float, default=0.0)
    ap.add_argument("--loss", default="bce", choices=['bce','focal','asl'])
    ap.add_argument("--asl-gamma-pos", type=float, default=0.0)
    ap.add_argument("--asl-gamma-neg", type=float, default=4.0)
    ap.add_argument("--asl-clip", type=float, default=0.05)
    ap.add_argument("--layer-decay", type=float, default=1.0)
    ap.add_argument("--rank-weight", type=float, default=0.0)
    ap.add_argument("--any-aux-weight", type=float, default=0.0)
    ap.add_argument("--view-consistency", type=float, default=0.0)
    ap.add_argument("--patient-smooth", type=float, default=0.0)
    ap.add_argument("--patient-smooth-c1", type=float, default=-1.0,
                    help=">=0 时类1单独用该系数；-1=跟随 --patient-smooth")
    ap.add_argument("--head-type", default="lse", choices=['lse','query','region'])
    ap.add_argument("--lse-alpha", type=float, default=0.0, help="混合 LSE 池化权重（0=纯平均）")
    ap.add_argument("--extra-channel", default="", choices=["", "asym"],
                    help="额外输入通道：asym=|I−fliplr(I)|，首层新通道零初始化")
    ap.add_argument("--init-weights", default="", help="外部主干权重（只加载 backbone.*）")
    ap.add_argument("--label-blend-48", type=float, default=0.0,
                    help="训练目标在类4↔类8之间互混的权重（0=关闭）")
    ap.add_argument("--lse-tau", type=float, default=1.0)
    ap.add_argument("--cache-dir", default="", help="预处理变体缓存目录（CLAHE/裁剪），空=默认 cache/")
    # 几何保护实验：类 1/6 的判据是心胸比与纵隔宽度，类 2/9 依赖肺尖与肋膈角，
    # 训练时的随机裁剪（面积 0.75–1.0、比例 0.85–1.18）会切掉这些外围证据。
    ap.add_argument("--no-crop", action="store_true",
                    help="训练时不裁剪，保留完整画幅（默认关闭=沿用现有裁剪）")
    ap.add_argument("--ratio-min", type=float, default=0.85, help="裁剪宽高比下界（默认 0.85）")
    ap.add_argument("--ratio-max", type=float, default=1.18, help="裁剪宽高比上界（默认 1.18）")
    # ---- 十类别独立增强的辅助分支（默认全关，主路径不变）----
    ap.add_argument("--aux-branch", action="store_true",
                    help="启用辅助视图分支（每图 25%% 概率、先抽图后均匀抽类别）")
    ap.add_argument("--aux-frac", type=float, default=0.25, help="辅助视图比例（默认 0.25）")
    ap.add_argument("--aux-weight", type=float, default=0.20, help="辅助项损失权重（默认 0.20）")
    ap.add_argument("--aux-policy", default="perclass", choices=["perclass", "uniform"],
                    help="perclass=C 组十类策略；uniform=B 组统一温和策略")
    ap.add_argument("--aux-policy-json", default="", help="策略 JSON 路径（空=用 04_文档 草案）")
    ap.add_argument("--jpg-cache", action="store_true",
                    help="从 cache_jpg/ 读 JPEG 缓存（上传带宽受限的远程训练）")
    ap.add_argument("--fold-csv", default="",
                    help="外部折表(Study_id,fold)，与 --train-folds/--valid-fold 配合")
    ap.add_argument("--train-folds", default="0,1,2")
    ap.add_argument("--valid-fold", type=int, default=3)
    ap.add_argument("--cv-folds", type=int, default=0)
    ap.add_argument("--cv-fold", type=int, default=0)
    ap.add_argument("--chronic-min", type=int, default=0)
    ap.add_argument("--chronic-factor", type=float, default=1.0)
    ap.add_argument("--oversample-view", default="")
    ap.add_argument("--oversample-factor", type=float, default=2.0)
    ap.add_argument("--view-filter", default="")
    ap.add_argument("--use-view-input", action="store_true",
                    help="把采集体位 one-hot 拼到分类头输入（针对 AP 便携片这一弱项）")
    return ap.parse_args()


def main():
    a = parse_args()
    cfg = TrainConfig(**{k: v for k, v in vars(a).items()
                         if k in TrainConfig.__dataclass_fields__})
    train(cfg)


if __name__ == "__main__":
    main()
