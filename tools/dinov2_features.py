"""最小 DINOv2 ViT-B/14 实现 + 权重加载（只做特征提取，不训练）。

为什么自己实现
--------------
timm 的 HF 入口在本机不可用（`huggingface_hub.hf_api` 缺 httpcore2），
官方 `torch.hub` 的下载域名对 HEAD 返回 403。权重本身能直接下（HF 镜像与官方源各 330.3 MB，
两者大小一致），键名是标准 HuggingFace 格式，因此这里用 ~80 行实现前向。

结构（ViT-B/14）
    patch_embeddings: Conv2d(3,768,14,14)
    cls_token, position_embeddings(1, 1+256, 768)  ← 518/14=37 → 37²=1369，248²/14²=256
    12 × Block( LayerNorm → Attention(qkv 分开) → LayerScale → Dense → LayerScale
                LayerNorm → MLP(fc1 3072, GELU, fc2) → LayerScale )
    layernorm

    python tools/dinov2_features.py --check          # 只做加载与形状自检
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn

ROOT = r"E:\挑战赛"
WEIGHTS = os.path.join(ROOT, ".torch_home", "dinov2", "dinov2_vitb14.safetensors")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


class Block(nn.Module):
    def __init__(self, dim: int = 768, n_heads: int = 12, mlp: int = 3072):
        super().__init__()
        self.n_heads = n_heads
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.ls1 = nn.Parameter(torch.zeros(dim))
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.fc1 = nn.Linear(dim, mlp)
        self.fc2 = nn.Linear(mlp, dim)
        self.ls2 = nn.Parameter(torch.zeros(dim))
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        h = self.norm1(x)
        q = self.q(h).reshape(B, N, self.n_heads, C // self.n_heads).transpose(1, 2)
        k = self.k(h).reshape(B, N, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = self.v(h).reshape(B, N, self.n_heads, C // self.n_heads).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (C // self.n_heads) ** -0.5
        att = att.softmax(dim=-1)
        o = (att @ v).transpose(1, 2).reshape(B, N, C)
        x = x + self.ls1 * self.proj(o)
        h = self.norm2(x)
        x = x + self.ls2 * self.fc2(self.act(self.fc1(h)))
        return x


class DINOv2ViT(nn.Module):
    """DINOv2 ViT-B/14（无 register），只输出 CLS 特征。"""

    def __init__(self, dim: int = 768, depth: int = 12, patch: int = 14, img: int = 448):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.n = (img // patch) ** 2
        self.pos = nn.Parameter(torch.zeros(1, 1 + self.n, dim))
        self.blocks = nn.ModuleList([Block(dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.proj(x).flatten(2).transpose(1, 2)           # B, n, C
        x = torch.cat([self.cls.expand(B, -1, -1), x], dim=1)
        x = x + self.pos[:, : x.shape[1]]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]                                        # CLS


def load_dinov2(img: int = 448, path: str = WEIGHTS) -> DINOv2ViT:
    from safetensors.torch import load_file
    sd = load_file(path)
    model = DINOv2ViT(img=img)
    # 位置编码：官方是 518×518（37×37 patch），这里按目标尺寸插值到 1+n
    pos_src = sd["embeddings.position_embeddings"]            # 1, 1+1369, 768
    want = model.pos.shape[1]
    if pos_src.shape[1] != want:
        cls_pos, patch_pos = pos_src[:, :1], pos_src[:, 1:]
        side_src = int(round(patch_pos.shape[1] ** 0.5))
        side_dst = int(round((want - 1) ** 0.5))
        grid = patch_pos.reshape(1, side_src, side_src, -1).permute(0, 3, 1, 2)
        grid = torch.nn.functional.interpolate(grid.float(), size=(side_dst, side_dst),
                                               mode="bicubic", align_corners=False)
        patch_pos = grid.permute(0, 2, 3, 1).reshape(1, side_dst * side_dst, -1)
        pos = torch.cat([cls_pos, patch_pos], dim=1).to(pos_src.dtype)
    else:
        pos = pos_src
    mapping = {
        "proj.weight": sd["embeddings.patch_embeddings.projection.weight"],
        "proj.bias": sd["embeddings.patch_embeddings.projection.bias"],
        "cls": sd["embeddings.cls_token"],
        "pos": pos,
        "norm.weight": sd["layernorm.weight"],
        "norm.bias": sd["layernorm.bias"],
    }
    for i in range(len(model.blocks)):
        p = f"encoder.layer.{i}."
        mapping[f"blocks.{i}.norm1.weight"] = sd[p + "norm1.weight"]
        mapping[f"blocks.{i}.norm1.bias"] = sd[p + "norm1.bias"]
        mapping[f"blocks.{i}.q.weight"] = sd[p + "attention.attention.query.weight"]
        mapping[f"blocks.{i}.q.bias"] = sd[p + "attention.attention.query.bias"]
        mapping[f"blocks.{i}.k.weight"] = sd[p + "attention.attention.key.weight"]
        mapping[f"blocks.{i}.k.bias"] = sd[p + "attention.attention.key.bias"]
        mapping[f"blocks.{i}.v.weight"] = sd[p + "attention.attention.value.weight"]
        mapping[f"blocks.{i}.v.bias"] = sd[p + "attention.attention.value.bias"]
        mapping[f"blocks.{i}.proj.weight"] = sd[p + "attention.output.dense.weight"]
        mapping[f"blocks.{i}.proj.bias"] = sd[p + "attention.output.dense.bias"]
        mapping[f"blocks.{i}.ls1"] = sd[p + "layer_scale1.lambda1"]
        mapping[f"blocks.{i}.norm2.weight"] = sd[p + "norm2.weight"]
        mapping[f"blocks.{i}.norm2.bias"] = sd[p + "norm2.bias"]
        mapping[f"blocks.{i}.fc1.weight"] = sd[p + "mlp.fc1.weight"]
        mapping[f"blocks.{i}.fc1.bias"] = sd[p + "mlp.fc1.bias"]
        mapping[f"blocks.{i}.fc2.weight"] = sd[p + "mlp.fc2.weight"]
        mapping[f"blocks.{i}.fc2.bias"] = sd[p + "mlp.fc2.bias"]
        mapping[f"blocks.{i}.ls2"] = sd[p + "layer_scale2.lambda1"]
    own = model.state_dict()
    missing = [k for k in own if k not in mapping]
    bad = [k for k, v in mapping.items() if tuple(v.shape) != tuple(own[k].shape)]
    if missing or bad:
        raise SystemExit(f"[error] 权重映射不完整 missing={missing[:5]} bad={bad[:5]}")
    model.load_state_dict({k: v.float() for k, v in mapping.items()}, strict=True)
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--size", type=int, default=448)
    a = ap.parse_args()
    m = load_dinov2(img=a.size)
    n_par = sum(p.numel() for p in m.parameters())
    print(f"  加载成功：{n_par/1e6:.1f}M 参数，输入 {a.size}，patch {m.patch}")
    with torch.no_grad():
        x = torch.randn(1, 3, a.size, a.size)
        y = m(x)
    print(f"  前向输出: {tuple(y.shape)}  均值 {y.mean():+.4f}  范数 {y.norm():.3f}")
    print(f"  是否有限值: {bool(torch.isfinite(y).all())}")


if __name__ == "__main__":
    main()
