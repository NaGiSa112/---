"""T1 异构骨干上卡前的接口 / 显存 / 吞吐自检（本机 4060 Laptop 8 GB）。

检查四件事：
1. 预训练能加载（走 HF_ENDPOINT 镜像或 CXR_WEIGHTS_DIR 本地文件）；
2. 灰度化正确：同一个灰度图复制成三通道 vs 直接一通道，输出应一致
   （首层 RGB 卷积按通道求和，所以两者在数学上等价）；
3. 空间特征图尺寸 + LSE 头能吃下（lse_alpha>0 走 `backbone.features`）；
4. 指定分辨率 / 物理批量的峰值显存与训练吞吐（含 AdamW 更新）。

    python tools/t1_backbone_smoke.py --arch efficientvit_b1.r288_in1k --img 384 --bs 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cxr.models import _adapt_first_conv, build_model  # noqa: E402


def stem_equivalence(arch: str, img: int) -> float:
    """首层 RGB 卷积按通道求和 -> 灰度，数学上应严格等价。

    直接比较「未改动的 3 通道 stem 吃 gray.repeat(3)」与「改成一通道的 stem 吃 gray」，
    避开分类头随机初始化带来的差异。
    """
    import timm
    import torch.nn as nn
    net = timm.create_model(arch, pretrained=True, num_classes=0).eval()
    conv3 = next(m for m in net.modules() if isinstance(m, nn.Conv2d) and m.in_channels == 3)
    conv1 = _adapt_first_conv(conv3, 1).eval()
    torch.manual_seed(0)
    g = torch.rand(1, 1, img, img)
    with torch.no_grad():
        return (conv3(g.repeat(1, 3, 1, 1)) - conv1(g)).abs().max().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="efficientvit_b1.r288_in1k")
    ap.add_argument("--img", type=int, default=384)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--amp", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    dev = torch.device("cuda")
    print(f"[smoke] {a.arch} img={a.img} bs={a.bs} amp={a.amp} "
          f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    model = build_model(a.arch, pretrained=True, drop_path=0.0, dropout=0.2,
                        lse_alpha=0.25, lse_tau=1.0, head_type="lse", in_ch=1,
                        img_size=0).to(dev)
    n_param = sum(p.numel() for p in model.parameters())
    bb_param = sum(p.numel() for p in model.backbone.parameters())
    print(f"[smoke] params total {n_param/1e6:.3f}M  backbone {bb_param/1e6:.3f}M  "
          f"fp32 size {n_param*4/2**20:.2f} MB", flush=True)

    # ---- 1. 灰度等价性（首层卷积层面） ----
    torch.manual_seed(0)
    g = torch.rand(1, 1, a.img, a.img, device=dev)
    model.eval()
    with torch.no_grad():
        o1 = model(g)
    d = stem_equivalence(a.arch, a.img)
    print(f"[smoke] stem grayscale-equiv max|Δ| = {d:.3e}  (应≈0)", flush=True)
    print(f"[smoke] logits {tuple(o1.shape)} range [{o1.min():.3f}, {o1.max():.3f}]", flush=True)

    # ---- 2. 空间特征形状 ----
    with torch.no_grad():
        f = model._spatial_features(g)
    print(f"[smoke] spatial features {tuple(f.shape)}", flush=True)

    # ---- 3. 训练步：显存 + 吞吐 ----
    model.train()
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 1e-4},
        {"params": model.head.parameters(), "lr": 1e-3},
    ], weight_decay=0.1)
    lossf = torch.nn.BCEWithLogitsLoss()
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": None}[a.amp]
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)

    x = torch.rand(a.bs, 1, a.img, a.img, device=dev)
    y = (torch.rand(a.bs, 10, device=dev) > 0.5).float()
    torch.cuda.reset_peak_memory_stats()
    t_all = time.time()
    for i in range(a.warmup + a.steps):
        if i == a.warmup:
            torch.cuda.synchronize()
            t0 = time.time()
            torch.cuda.reset_peak_memory_stats()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            loss = lossf(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    dt = time.time() - t0
    ips = a.bs * a.steps / dt
    peak_a = torch.cuda.max_memory_allocated() / 2**20
    peak_r = torch.cuda.max_memory_reserved() / 2**20
    total_wall = time.time() - t_all
    print(f"[smoke] {a.steps} steps in {dt:.2f}s -> {dt/a.steps*1000:.1f} ms/step, "
          f"{ips:.1f} img/s", flush=True)
    print(f"[smoke] peak allocated {peak_a:.0f} MiB | reserved {peak_r:.0f} MiB | "
          f"device total {torch.cuda.get_device_properties(0).total_memory/2**20:.0f} MiB",
          flush=True)
    # 12 轮 24,194 图的纯训练小时数（不含验证/存盘）
    hours = 24194 * 12 / ips / 3600
    print(f"[smoke] 12 epochs x 24,194 imgs ≈ {hours:.2f} h (纯训练，按实测吞吐)", flush=True)

    if a.out:
        rec = dict(arch=a.arch, img=a.img, bs=a.bs, amp=a.amp,
                   params=n_param, backbone_params=bb_param,
                   grayscale_maxdiff=d, spatial=list(f.shape),
                   ms_per_step=dt / a.steps * 1000, img_per_s=ips,
                   peak_allocated_mib=peak_a, peak_reserved_mib=peak_r,
                   est_12ep_hours=hours, gpu=torch.cuda.get_device_name(0),
                   torch=torch.__version__)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=2, ensure_ascii=False)
        print(f"[smoke] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
