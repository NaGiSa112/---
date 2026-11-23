"""把官方 ConvNeXt-Tiny ImageNet-22K 权重转成本项目模型的 state_dict（精确映射版）。

结构对照
--------
官方（facebookresearch/ConvNeXt，`convnext_tiny_22k_1k_224.pth`）
    主干按"先所有 downsample，再所有 stage"排列：
    downsample_layers.{0..3}.{0,1}  stages.{0..3}.{i}.{gamma,dwconv,norm,pwconv1,pwconv2}
    norm.{weight,bias}  head.{weight,bias}

本项目（torchvision ConvNeXt 骨架，单通道 stem）
    主干是**一个** Sequential：stem → stage0 的 3 个 block → downsample → stage1 的 3 个 block
    → downsample → stage2 的 9 个 block → downsample → stage3 的 3 个 block
    backbone.0.<seq>.<blk>.{layer_scale, block.0/2/3/5}
    非 stem 的 downsample 是 Sequential(LayerNorm(C_in), Conv2d(C_in, C_out, 2, 2))

映射规则
--------
    downsample_layers.0.0 (Conv2d 96,3,4,4) → backbone.0.0.0   （3 通道求和为 1 通道）
    downsample_layers.0.1 (LayerNorm 96)    → backbone.0.0.1
    downsample_layers.{j}.0 (LayerNorm)     → backbone.0.{2j}.0
    downsample_layers.{j}.1 (Conv2d)        → backbone.0.{2j}.1        j=1,2,3
    stages.{s}.{i}.gamma                    → backbone.0.{off+s}.{i}.layer_scale（reshape 成 C,1,1）
    stages.{s}.{i}.dwconv                   → ...block.0
    stages.{s}.{i}.norm                     → ...block.2
    stages.{s}.{i}.pwconv1                  → ...block.3
    stages.{s}.{i}.pwconv2                  → ...block.5
    off = [1, 3, 5, 7][s]（stage 0/1/2/3 的起始序号）
    norm.* / head.*                         不迁移（本项目 final_norm=False，分类头是 10 类）

    python tools/convert_convnext22k.py
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

ROOT = r"E:\挑战赛"
sys.path.insert(0, os.path.join(ROOT, "src"))

from cxr.models import build_model  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DEFAULT_CKPT = os.path.join(ROOT, ".torch_home", "pretrained22k", "convnext_tiny_22k_1k_224.pth")
DEFAULT_OUT = os.path.join(ROOT, "runs", "init22k", "tiny22k_adapted.pt")
NBLK = {0: 3, 1: 3, 2: 9, 3: 3}         # ConvNeXt-Tiny 各 stage 的 block 数
BLOCKMAP = {"dwconv": "block.0", "norm": "block.2", "pwconv1": "block.3", "pwconv2": "block.5"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--in-ch", type=int, default=1, choices=[1, 2])
    a = ap.parse_args()

    raw = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    src = raw.get("model", raw)
    model = build_model("convnext_tiny", pretrained=False, drop_path=0.0, in_ch=a.in_ch)
    tgt = model.state_dict()
    out: dict[str, torch.Tensor] = {}
    used: set[str] = set()

    def put(dst_key: str, tensor: torch.Tensor, note: str = "") -> None:
        if dst_key not in tgt:
            raise SystemExit(f"[error] 目标里没有键 {dst_key}")
        want = tuple(tgt[dst_key].shape)
        if tuple(tensor.shape) != want:
            if tensor.numel() != int(torch.tensor(want).prod()):
                raise SystemExit(f"[error] {dst_key}: {tuple(tensor.shape)} vs {want}")
            tensor = tensor.reshape(want)
        out[dst_key] = tensor.clone()
        if note:
            print(f"    {note}")

    # ---- stem ----
    w = src["downsample_layers.0.0.weight"]              # (96,3,4,4)
    stem = torch.zeros(tuple(tgt["backbone.0.0.0.weight"].shape), dtype=w.dtype)
    stem[:, :1] = w.sum(dim=1, keepdim=True)
    put("backbone.0.0.0.weight", stem, f"stem conv {tuple(w.shape)} → 3 通道求和")
    put("backbone.0.0.0.bias", src["downsample_layers.0.0.bias"])
    put("backbone.0.0.1.weight", src["downsample_layers.0.1.weight"])
    put("backbone.0.0.1.bias", src["downsample_layers.0.1.bias"])

    # ---- 非 stem 的 downsample ----
    for j in (1, 2, 3):
        put(f"backbone.0.{2*j}.0.weight", src[f"downsample_layers.{j}.0.weight"])
        put(f"backbone.0.{2*j}.0.bias", src[f"downsample_layers.{j}.0.bias"])
        put(f"backbone.0.{2*j}.1.weight", src[f"downsample_layers.{j}.1.weight"])
        put(f"backbone.0.{2*j}.1.bias", src[f"downsample_layers.{j}.1.bias"])

    # ---- 各 stage 的 block ----
    for s in range(4):
        for i in range(NBLK[s]):
            base = f"stages.{s}.{i}"
            # 主干结构：backbone.0 = [stem, stage0, ds1, stage1, ds2, stage2, ds3, stage3]
            dst = f"backbone.0.{2 * s + 1}.{i}"
            g = src[f"{base}.gamma"]
            put(f"{dst}.layer_scale", g.reshape(tgt[f"{dst}.layer_scale"].shape))
            for k_src, k_dst in BLOCKMAP.items():
                put(f"{dst}.{k_dst}.weight", src[f"{base}.{k_src}.weight"])
                put(f"{dst}.{k_dst}.bias", src[f"{base}.{k_src}.bias"])

    # ---- 分类头保持随机初始化（10 类，不迁移 1000 类权重）----
    for k in tgt:
        if k.startswith("head."):
            out[k] = tgt[k].clone()

    missing = [k for k in tgt if k not in out]
    if missing:
        raise SystemExit(f"[error] 未填充的键 {len(missing)} 个，例如 {missing[:5]}")

    model.load_state_dict(out, strict=True)
    n_src = len([k for k in src if not k.startswith(("head.", "norm."))])
    print(f"  官方主干张量 {n_src} 个 → 目标 {len(out)} 个键，strict=True 通过")
    print(f"  未迁移：norm.*（本项目 final_norm=False）、head.*（{tuple(src['head.weight'].shape)} → 10 类）")

    model.eval()
    with torch.no_grad():
        y = model(torch.randn(1, a.in_ch, 128, 128))
    print(f"  前向自检: 输出 {tuple(y.shape)}，均值 {y.mean().item():+.4f}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save({"model": out, "arch": "convnext_tiny", "in_ch": a.in_ch,
                "source": os.path.basename(a.ckpt),
                "note": "official ConvNeXt-Tiny IN22k-ft1k, final norm & head dropped"},
               a.out)
    print(f"  [wrote] {os.path.relpath(a.out, ROOT)}  {os.path.getsize(a.out)/1e6:.2f} MB")


if __name__ == "__main__":
    main()
