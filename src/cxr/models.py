"""Model factory: ImageNet-pretrained backbones adapted to 1-channel chest X-ray."""
from __future__ import annotations

import math
import os

import torch
import torch.nn as nn

from .config import NUM_CLASSES


def _adapt_first_conv(conv: nn.Conv2d, in_ch: int = 1) -> nn.Conv2d:
    """Collapse the 3-channel ImageNet stem onto grayscale, optionally adding extra channels.

    in_ch=1 reproduces the historical behaviour exactly. Extra channels start with ZERO
    weights, so a model built with in_ch=2 is step-0 identical to the 1-channel model and
    can only deviate if the extra input actually helps.
    """
    w = conv.weight.data.sum(dim=1, keepdim=True)
    out = nn.Conv2d(in_ch, conv.out_channels, kernel_size=conv.kernel_size,
                    stride=conv.stride, padding=conv.padding, bias=conv.bias is not None)
    new_w = torch.zeros(conv.out_channels, in_ch, *w.shape[2:], dtype=w.dtype,
                        device=w.device)
    new_w[:, :1] = w
    out.weight.data.copy_(new_w)
    if conv.bias is not None:
        out.bias.data.copy_(conv.bias.data)
    return out


class RegionHead(nn.Module):
    """Anatomy-aware pooling: pool a fixed g x g grid and let the classifier read it.

    Chest findings live in characteristic places (cardiomegaly in the heart shadow,
    effusion in the costophrenic angles), but a global average throws that layout away
    and the learned-query alternative (QueryHead) has too many parameters for 18k
    studies. A fixed grid keeps the layout with only g*g*C*10 weights (69k at g=3),
    which is small enough to train on this data.
    """

    def __init__(self, dim: int, num_classes: int, grid: int = 3, dropout: float = 0.2,
                 with_global: bool = True):
        super().__init__()
        self.grid = grid
        self.with_global = with_global
        cells = grid * grid + (1 if with_global else 0)
        self.pool = nn.AdaptiveAvgPool2d((grid, grid))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim * cells, num_classes))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        r = self.pool(feat).flatten(1)                       # B x (g*g*C)
        if self.with_global:
            r = torch.cat([feat.mean(dim=(2, 3)), r], dim=1)
        return self.head(r)


class QueryHead(nn.Module):
    """Per-class attention pooling: one learned query per class attends over the map.

    This is the Ml-Decoder idea (Ridnik et al.) reduced to its essentials: instead of a
    fixed pooling rule (average or LSE), each class gets its own query vector and reads
    whatever spatial evidence it needs. CheXFusion reports +0.008 mAP on CXR-LT from
    swapping a GAP head for this, at the cost of a few hundred thousand parameters.
    """

    def __init__(self, dim: int, num_classes: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.queries = nn.Parameter(torch.randn(num_classes, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        b = feat.shape[0]
        tokens = self.norm(feat.flatten(2).transpose(1, 2))   # B x HW x C
        q = self.queries.unsqueeze(0).expand(b, -1, -1)
        out, _ = self.attn(q, tokens, tokens, need_weights=False)
        return out + self.ffn(out)                            # B x num_classes x C


class CXRNet(nn.Module):
    def __init__(self, arch: str = "convnext_tiny", num_classes: int = NUM_CLASSES,
                 pretrained: bool = True, drop_path: float = 0.1, dropout: float = 0.2,
                 n_aux: int = 0, n_view: int = 0, final_norm: bool = False,
                 lse_alpha: float = 0.0, lse_tau: float = 1.0, head_type: str = "lse",
                 in_ch: int = 1, img_size: int = 0):
        super().__init__()
        self.arch = arch
        self.n_aux = n_aux
        self.n_view = n_view
        self.final_norm = final_norm
        self.head_type = head_type
        # Global average pooling dilutes a lesion that only lights up in a few of the
        # 16x16 spatial positions. Mixing in a log-sum-exp aggregate (as in ChestX-ray8)
        # keeps the global evidence while letting a strong local response through.
        # lse_alpha=0 reproduces the plain average-pooling path exactly, and because this
        # adds NO parameters, checkpoints stay interchangeable between the two modes.
        self.lse_alpha = float(lse_alpha)
        self.lse_tau = float(lse_tau)
        self.p_drop = float(dropout)
        self.backbone, feat_dim = _build_backbone(arch, pretrained, drop_path, final_norm,
                                                  in_ch=in_ch, img_size=img_size)
        # n_view > 0 feeds the one-hot acquisition view (AP/PA/LATERAL/LL/other) into
        # the classifier. The view is otherwise only inferable from image appearance,
        # which is exactly where the "portable AP => abnormal" shortcut lives: AP films
        # have a 77% disease rate vs 38% for laterals, and single portable AP studies
        # carry 51.8% of the online score with only ~0.74 AUC.
        head_in = feat_dim + (n_view if n_view else 0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(head_in, num_classes))
        # Query head: each class owns a query that attends over the feature map, so the
        # pooling rule is learned per class instead of fixed (avg / LSE).
        self.query_head = (QueryHead(feat_dim, num_classes, dropout=dropout)
                           if head_type == "query" else None)
        self.region_head = (RegionHead(feat_dim, num_classes, dropout=dropout)
                            if head_type == "region" else None)
        self.query_proj = (nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_dim, 1))
                           if head_type == "query" else None)
        self.aux_head = nn.Linear(feat_dim, n_aux) if n_aux else None
        # NOTE: an "any finding" auxiliary head was tried here and measured -0.0069 on
        # fold4, so it is gone. It also changed the state dict of every LSE model, which
        # broke loading of checkpoints trained before it existed -- keep this constructor
        # backward compatible.
        self.any_head = None

    def forward(self, x, return_aux: bool = False, view=None):
        if self.region_head is not None and not self.n_view:
            feat = self._spatial_features(x)
            if self.training and self.p_drop > 0:
                keep = (torch.rand(feat.shape[0], feat.shape[1], 1, 1, device=feat.device)
                        >= self.p_drop).to(feat.dtype) / (1.0 - self.p_drop)
                feat = feat * keep
            return self.region_head(feat).to(feat.dtype)
        if self.query_head is not None and not self.n_view:
            feat = self._spatial_features(x)
            if self.training and self.p_drop > 0:
                keep = (torch.rand(feat.shape[0], feat.shape[1], 1, 1, device=feat.device)
                        >= self.p_drop).to(feat.dtype) / (1.0 - self.p_drop)
                feat = feat * keep
            cls_tokens = self.query_head(feat)                    # B x 10 x C
            return self.query_proj(cls_tokens).squeeze(-1).to(feat.dtype)
        if self.lse_alpha > 0 and not self.n_view:
            return self._forward_lse(x)
        f = self.backbone(x)
        if self.n_view:
            if view is None:
                raise ValueError("model was built with n_view>0 but no view tensor was given")
            v = torch.nn.functional.one_hot(view.long(), self.n_view).to(f.dtype)
            f = torch.cat([f, v], dim=1)
        out = self.head(f)
        if return_aux and self.aux_head is not None:
            return out, self.aux_head(f[:, :f.shape[1] - (self.n_view or 0)] if self.n_view else f)
        return out

    def _spatial_features(self, x):
        """Pre-pooling feature map as B x C x H x W.

        ConvNeXt exposes features as a Sequential, so element [0] is the body up to the
        last stage. Swin returns a channels-last map (B, H, W, C) from `features` and is
        not subscriptable, which is why LSE pooling needs this adapter rather than the
        ConvNeXt-specific `backbone[0]` call.
        """
        if isinstance(self.backbone, nn.Sequential):
            return self.backbone[0](x)
        feat = self.backbone.features(x)
        if feat.dim() == 4 and feat.shape[-1] > feat.shape[1]:
            feat = feat.permute(0, 3, 1, 2)             # B,H,W,C -> B,C,H,W
        return feat

    def _forward_lse(self, x):
        """Apply the classifier at every spatial position, then mix AVG and LSE.

        z_avg = mean(M),  z_lse = tau * (logsumexp(M/tau) - log(N))
        z = (1-alpha) * z_avg + alpha * z_lse,  except the last class which keeps z_avg
        (class 10 is the complement of the other nine, so its global judgement is kept).
        """
        feat = self._spatial_features(x)                # B x C x H x W (pre-pooling)
        if self.training and self.p_drop > 0:
            # Channel-wise dropout with one mask per sample, broadcast over space: this is
            # the spatial analogue of dropping the pooled vector, so the only change
            # against the baseline is the pooling rule itself.
            keep = (torch.rand(feat.shape[0], feat.shape[1], 1, 1, device=feat.device)
                    >= self.p_drop).to(feat.dtype) / (1.0 - self.p_drop)
            feat = feat * keep
        w, b = self.head[1].weight, self.head[1].bias     # 10 x C , 10
        m = torch.einsum("bchw,oc->bohw", feat.float(), w) + b.view(1, -1, 1, 1)
        n_sp = m.shape[2] * m.shape[3]
        z_avg = m.mean(dim=(2, 3))
        tau = self.lse_tau
        z_lse = tau * (torch.logsumexp(m / tau, dim=(2, 3)) - math.log(n_sp))
        z = (1.0 - self.lse_alpha) * z_avg + self.lse_alpha * z_lse
        z = torch.cat([z[:, :-1], z_avg[:, -1:]], dim=1)
        z = z.to(feat.dtype)
        if self.any_head is not None:
            self._last_any = self.any_head(z)
        return z


class _TimmSpatial(nn.Module):
    """把 timm 模型包成统一形态：forward 返回池化向量，.features(x) 返回 B×C×H×W 空间图。

    MetaFormer（CAFormer）与 SwinV2 的 forward_features 返回 NHWC，这里统一转成 BCHW，
    这样 CXRNet 的 LSE 头与 torchvision 分支可以共用同一段代码。
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        self.out_channels = _timm_out_channels(net)

    def features(self, x):
        f = self.net.forward_features(x)
        if f.dim() == 4 and f.shape[-1] > f.shape[1]:
            f = f.permute(0, 3, 1, 2)                 # NHWC -> BCHW
        return f

    def forward(self, x):
        f = self.features(x)
        return f.mean(dim=(2, 3))


def _timm_out_channels(net: nn.Module) -> int:
    for attr in ("num_features", "feature_dim", "embed_dim"):
        v = getattr(net, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    for _, m in net.named_modules():
        if isinstance(m, nn.Linear):
            return int(m.in_features)
    raise RuntimeError("cannot infer timm head features")


def _adapt_first_conv_any(model: nn.Module, in_ch: int = 1) -> str:
    """把模型里第一个 in_channels==3 的卷积改成 in_ch 通道（灰度或灰度+额外通道）。

    比按属性名硬编码更稳：CAFormer 的 SepStem、SwinV2 的 patch_embed 结构名都不同。
    """
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d) and mod.in_channels == 3:
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
            leaf = parts[-1]
            new = _adapt_first_conv(mod, in_ch)
            if leaf.isdigit():
                parent[int(leaf)] = new
            else:
                setattr(parent, leaf, new)
            return name
    raise RuntimeError("no 3-channel conv found for stem adaptation")


def _build_backbone(arch: str, pretrained: bool, drop_path: float, final_norm: bool = False,
                    in_ch: int = 1, img_size: int = 0):
    import torchvision as tv

    if final_norm and not arch.startswith("convnext"):
        raise ValueError("final_norm is currently supported only for ConvNeXt")
    if arch.startswith("convnext"):
        weights = getattr(tv.models, f"ConvNeXt_{arch.split('_')[1].capitalize()}_Weights", None)
        net = getattr(tv.models, arch)(weights=weights.DEFAULT if (pretrained and weights) else None)
        model = net.features
        model[0][0] = _adapt_first_conv(model[0][0], in_ch)
        feat_dim = net.classifier[-1].in_features
        _set_drop_path(model, drop_path)
        # Opt-in to preserve existing checkpoint keys and inference behavior.
        # Keep the pretrained LayerNorm2d BEFORE flattening, as in torchvision.
        body = (nn.Sequential(model, net.avgpool, net.classifier[0], nn.Flatten(1))
                if final_norm else
                nn.Sequential(model, nn.AdaptiveAvgPool2d(1), nn.Flatten(1)))
        return body, feat_dim

    if arch == "swin_t":
        net = tv.models.swin_t(weights=tv.models.Swin_T_Weights.DEFAULT if pretrained else None)
        net.features[0][0] = _adapt_first_conv(net.features[0][0], in_ch)
        feat_dim = net.head.in_features
        net.head = nn.Identity()
        if hasattr(net, "norm"):
            net.norm = nn.Identity()
        return net, feat_dim

    if arch.startswith("efficientnet_b"):
        net = getattr(tv.models, arch)(weights="DEFAULT" if pretrained else None)
        net.features[0][0] = _adapt_first_conv(net.features[0][0], in_ch)
        feat_dim = net.classifier[-1].in_features
        net.classifier = nn.Identity()
        return net, feat_dim

    if arch in ("resnet50", "resnet34", "resnet18"):
        net = getattr(tv.models, arch)(weights="DEFAULT" if pretrained else None)
        net.conv1 = _adapt_first_conv(net.conv1, in_ch)
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        return net, feat_dim

    if arch.startswith("densenet"):
        net = getattr(tv.models, arch)(weights="DEFAULT" if pretrained else None)
        net.features[0] = _adapt_first_conv(net.features[0], in_ch)
        feat_dim = net.classifier.in_features
        net.classifier = nn.Identity()
        return net, feat_dim

    if arch.startswith(("caformer", "swinv2", "convnextv2", "maxvit", "coatnet")):
        # timm 分支：MetaFormer / SwinV2 / ConvNeXtV2 等。预训练数据均为通用自然图像
        # （ImageNet-22K / SAIL 蒸馏 / IN1K），不使用任何医学权重。
        import timm
        kw = dict(pretrained=pretrained, num_classes=0, drop_path_rate=drop_path)
        # 云端常常连不上 huggingface.co（国内容器），但能连 modelscope.cn。
        # 设 CXR_WEIGHTS_DIR 后，从 <dir>/<arch>/model.safetensors 直接加载，完全不走网络。
        wdir = os.environ.get("CXR_WEIGHTS_DIR", "")
        if pretrained and wdir:
            for fn in ("model.safetensors", "pytorch_model.bin"):
                cand = os.path.join(wdir, arch.replace("/", "_"), fn)
                if os.path.exists(cand):
                    kw["pretrained_cfg_overlay"] = {"file": cand}
                    print(f"[backbone] 从本地权重加载 {os.path.basename(cand)} ← {cand}")
                    break
        # 并非所有 timm 架构都接受 img_size：MetaFormer 家族（CAFormer）没有位置编码，
        # 其 __init__ 直接拒绝 img_size（实测 TypeError）；SwinV2 则**必须**按目标尺寸构造，
        # 否则默认 256 会在 512/544 输入下报错。所以先试着带 img_size，不支持时再退回。
        net = None
        if img_size:
            try:
                net = timm.create_model(arch, img_size=int(img_size), **kw)
            except TypeError as exc:
                if "img_size" not in str(exc):
                    raise
                print(f"[backbone] {arch} 不接受 img_size，改用默认构造（训练时按实际输入前向）")
        if net is None:
            net = timm.create_model(arch, **kw)
        # 保险：SwinV2 等带有可变的输入尺寸属性，若 create_model 没吃进 img_size，
        # 这里再显式设置一次（否则前向时 input 512 vs model 256 会直接断言失败）。
        if img_size and hasattr(net, "set_input_size"):
            try:
                cur = getattr(getattr(net, "patch_embed", None), "img_size", None)
                if cur is None or int(cur[0]) != int(img_size):
                    net.set_input_size((int(img_size), int(img_size)))
                    print(f"[backbone] set_input_size -> {img_size}")
            except Exception as exc:                       # noqa: BLE001
                print(f"[backbone] set_input_size 失败（忽略）: {exc}")
        stem = _adapt_first_conv_any(net, in_ch)
        print(f"[backbone] timm {arch}  stem={stem}  in_ch={in_ch}  drop_path={drop_path}")
        return _TimmSpatial(net), _timm_out_channels(net)

    raise ValueError(f"unknown arch {arch}")


def _set_drop_path(module: nn.Module, rate: float):
    """Set the maximum drop probability, preserving ConvNeXt's depth ramp."""
    from torchvision.ops import StochasticDepth

    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"drop_path must be in [0, 1], got {rate}")
    blocks = [m for m in module.modules() if isinstance(m, StochasticDepth)]
    for i, block in enumerate(blocks):
        # torchvision stores the probability on StochasticDepth.p, not on
        # CNBlock.drop_path_rate. Updating the latter alone silently did nothing.
        block.p = rate * i / max(len(blocks) - 1, 1)
    for m in module.modules():
        if hasattr(m, "drop_path_rate"):
            m.drop_path_rate = rate


def build_model(arch: str, pretrained: bool = True, drop_path: float = 0.1, dropout: float = 0.2,
                n_aux: int = 0, n_view: int = 0, final_norm: bool = False,
                lse_alpha: float = 0.0, lse_tau: float = 1.0, head_type: str = "lse",
                in_ch: int = 1, img_size: int = 0):
    return CXRNet(arch, NUM_CLASSES, pretrained=pretrained, drop_path=drop_path,
                  dropout=dropout, n_aux=n_aux, n_view=n_view, final_norm=final_norm,
                  lse_alpha=lse_alpha, lse_tau=lse_tau, head_type=head_type, in_ch=in_ch,
                  img_size=img_size)


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        import copy
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for pe, pm in zip(self.ema.state_dict().values(), model.state_dict().values()):
            if pe.dtype.is_floating_point:
                pe.mul_(d).add_(pm.detach(), alpha=1 - d)
            else:
                pe.copy_(pm)
