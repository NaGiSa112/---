"""E0 接入验证（CPU 可跑部分）：CAFormer-S18 与 SwinV2-Tiny 能否接进我们的训练/推理栈。

检查项
------
1. 构造：能否建出模型（timm，pretrained=False 以免占网络）
2. stem：第一个 3 通道卷积能否改成单通道，且参数量变化符合预期
3. LSE：`_spatial_features` 是否返回 B×C×H×W，尺寸是否与 ConvNeXt 同构
4. 前向：一次前向能否输出 (B, 10)，且数值有限
5. 体积：fp32 权重字节数
6. drop_path：是否按传入 rate 生效

    python tools/e0_adapter_check.py
"""
from __future__ import annotations

import json
import os
import sys

import torch

ROOT = r"E:\挑战赛"
sys.path.insert(0, os.path.join(ROOT, "src"))

from cxr.models import build_model  # noqa: E402

OUT = os.path.join(ROOT, "reports", "e0_adapter_check.json")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

CASES = [
    ("convnext_tiny", 544),          # 基线：必须仍然工作
    ("caformer_s18.sail_in22k_ft_in1k", 544),
    ("swinv2_tiny_window16_256.ms_in1k", 512),
]


def main() -> None:
    results = []
    for arch, img in CASES:
        print(f"\n=== {arch}  @ {img}")
        rec = {"arch": arch, "img": img}
        try:
            m = build_model(arch, pretrained=False, drop_path=0.15, n_aux=0, n_view=0,
                            lse_alpha=0.30, lse_tau=1.0, head_type="lse", in_ch=1,
                            img_size=img)
            m.eval()
            n_par = sum(p.numel() for p in m.parameters())
            n_bytes = sum(p.numel() * 4 for p in m.parameters())
            rec["params_m"] = round(n_par / 1e6, 1)
            rec["weights_mb"] = round(n_bytes / 1e6, 1)
            print(f"  参数 {n_par/1e6:.1f}M   体积(fp32) {n_bytes/1e6:.1f} MB")

            # stem 是否已变成 1 通道
            first = None
            for name, mod in m.named_modules():
                if isinstance(mod, torch.nn.Conv2d):
                    first = (name, mod.in_channels, mod.out_channels)
                    break
            rec["first_conv"] = first
            print(f"  第一个卷积: {first}")
            rec["stem_ok"] = bool(first and first[1] == 1)

            x = torch.randn(2, 1, img, img)
            with torch.no_grad():
                feat = m._spatial_features(x)
                out = m(x)
            rec["feat_shape"] = list(feat.shape)
            rec["out_shape"] = list(out.shape)
            rec["finite"] = bool(torch.isfinite(out).all())
            print(f"  空间特征 {tuple(feat.shape)}   输出 {tuple(out.shape)}   "
                  f"有限值={rec['finite']}")

            # drop_path 是否生效
            dps = [getattr(mm, "drop_path_rate", None) for mm in m.modules()]
            dps = [d for d in dps if isinstance(d, float)]
            rec["drop_path_max"] = max(dps) if dps else None
            print(f"  drop_path 生效值: {rec['drop_path_max']}")

            ok = (rec["stem_ok"] and rec["out_shape"] == [2, 10] and rec["finite"])
            rec["pass"] = bool(ok)
            print(f"  → {'✅ 通过' if ok else '❌ 未通过'}")
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            rec["pass"] = False
            print(f"  ❌ {rec['error']}")
        results.append(rec)

    json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n  汇总")
    for r in results:
        flag = "✅" if r.get("pass") else "❌"
        print(f"   {flag} {r['arch'][:44]:44s} {r.get('params_m', '—')}M  "
              f"feat={r.get('feat_shape', '—')}")
    print(f"\n  [wrote] {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
