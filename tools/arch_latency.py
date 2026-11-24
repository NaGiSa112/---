"""实测候选主干在 544 输入下的单张延迟（随机初始化，只为测算力）。

    python tools/arch_latency.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

ROOT = r"E:\挑战赛"
OUT = os.path.join(ROOT, "reports", "arch_latency_2026-11-24.json")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

CANDS = [
    ("convnext_tiny.fb_in22k_ft_in1k", 544),      # 我们现用
    ("caformer_s18.sail_in22k_ft_in1k", 544),     # 推荐候选
    ("caformer_s36.sail_in22k_ft_in1k_384", 384),
    ("convnextv2_tiny.fcmae_ft_in22k_in1k", 544),
    ("maxvit_tiny_tf_224.in1k", 544),
    ("coatnet_1_rw_224.sw_in1k", 544),
    ("swinv2_tiny_window16_256.ms_in1k", 544),
    ("efficientvit_b1.r288_in1k", 544),
]


def main() -> None:
    import timm
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    print(f"  设备: {dev}")
    print(f"  {'arch':44s} {'参数':>7s} {'输入':>5s} {'延迟(ms)':>9s} {'显存(MB)':>9s}")
    for name, img in CANDS:
        try:
            m = timm.create_model(name, pretrained=False, num_classes=10)
            m = m.to(dev).eval()
            n_par = sum(p.numel() for p in m.parameters()) / 1e6
            x = torch.randn(1, 3, img, img, device=dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for _ in range(3):
                    m(x)
                torch.cuda.synchronize()
                ts = []
                for _ in range(15):
                    t0 = time.perf_counter()
                    m(x)
                    torch.cuda.synchronize()
                    ts.append((time.perf_counter() - t0) * 1000)
            ms = float(sorted(ts)[len(ts) // 2])
            mem = torch.cuda.max_memory_allocated() / 1e6
            torch.cuda.reset_peak_memory_stats()
            rows.append({"arch": name, "img": img, "params_m": round(n_par, 1),
                         "ms": round(ms, 1), "mem_mb": round(mem, 0)})
            print(f"  {name:44s} {n_par:6.1f}M {img:5d} {ms:9.1f} {mem:9.0f}")
            del m
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            print(f"  {name:44s} 失败: {type(e).__name__}: {str(e)[:40]}")
    json.dump(rows, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n  [wrote] {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
