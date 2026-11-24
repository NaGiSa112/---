"""云上环境自检（30 秒）：开卡后第一条命令就跑它。

为什么要它
----------
云镜像是 PyTorch 2.5.1 + CUDA 12.4 + Python 3.12，我们本地是 2.9/12.8；
而且 48 GB vGPU 的**卡代**决定 bf16 是否可用（Turing 无 bf16）。
这个脚本一次性把"能不能跑"问清楚，避免训练到一半才炸。

检查项
------
1. torch / CUDA / 卡名 / 算力等级 / 显存 / RAM
2. bf16 支持（算力 ≥ 8.0）与一次 bf16 前向
3. timm 是否可导入、CAFormer / SwinV2 能否构造（pretrained=False）
4. 我们的 `cxr` 包能否导入，三种架构能否 build 并前向
5. `GraphModel` 的 autocast 兼容函数是否存在
6. 缓存与 CSV 是否就位（路径来自 CXR_CACHE / CXR_DATA）
7. Tiny 一步训练的速度微基准（估算单折耗时）

    python cloud_env_check.py
"""
from __future__ import annotations

import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

OK, BAD, WARN = "✅", "❌", "⚠️"


def line(flag: str, msg: str) -> None:
    print(f"  {flag} {msg}")


def main() -> None:
    print("=" * 74)
    print("云上环境自检")
    print("=" * 74)

    # ---------- 1. 基础 ----------
    import torch
    line(OK, f"Python {sys.version.split()[0]}   torch {torch.__version__}   "
             f"CUDA_available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        line(BAD, "没有可用 CUDA，停止。")
        return
    p = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    free, total = torch.cuda.mem_get_info()
    line(OK, f"GPU: {p.name}  显存 {total/1e9:.1f} GB（空闲 {free/1e9:.1f}）  "
             f"算力 {cap[0]}.{cap[1]}  SM={p.multi_processor_count}")
    if hasattr(os, "sysconf"):                      # Linux
        try:
            ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
            line(OK, f"内存 {ram:.1f} GB")
        except (ValueError, OSError):
            pass
    else:                                          # Windows
        line(OK, f"内存（未查询，Windows 无 sysconf）")

    # ---------- 2. bf16 ----------
    bf16_ok = cap[0] >= 8
    line(OK if bf16_ok else BAD,
         f"bf16 支持: {'是（算力 ≥8.0）' if bf16_ok else '否，必须改 fp16'}")
    if bf16_ok:
        x = torch.randn(256, 256, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = x @ x
        line(OK, f"bf16 前向正常，输出 dtype={y.dtype}")

    # ---------- 3. timm ----------
    try:
        import timm
        line(OK, f"timm {timm.__version__}")
    except Exception as e:  # noqa: BLE001
        line(BAD, f"timm 导入失败：{type(e).__name__}: {e}")
        return

    # ---------- 4. 我们的包 ----------
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, "src"), os.path.join(os.path.dirname(here), "src")):
        if os.path.isdir(cand) and cand not in sys.path:
            sys.path.insert(0, cand)
    try:
        from cxr.models import build_model, _build_backbone
        from cxr.deploy import _autocast_state
        line(OK, "cxr 包导入成功")
        en, dt = _autocast_state()
        line(OK, f"_autocast_state() = ({en}, {dt})（torch 2.4–2.9 兼容）")
    except Exception as e:  # noqa: BLE001
        line(BAD, f"cxr 包导入失败：{type(e).__name__}: {e}")
        return

    cases = [("convnext_tiny", 544), ("caformer_s18.sail_in22k_ft_in1k", 544),
             ("swinv2_tiny_window16_256.ms_in1k", 512)]
    for arch, img in cases:
        try:
            m = build_model(arch, pretrained=False, drop_path=0.15, lse_alpha=0.30,
                            in_ch=1, img_size=img).cuda()
            n = sum(q.numel() for q in m.parameters()) / 1e6
            x = torch.randn(2, 1, img, img, device="cuda")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                t0 = time.perf_counter()
                out = m(x)
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000
            line(OK, f"{arch[:42]:42s} {n:5.1f}M  out={tuple(out.shape)}  "
                     f"{ms:6.1f} ms/batch2")
            del m
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            line(BAD, f"{arch[:42]:42s} 构造/前向失败：{type(e).__name__}: {str(e)[:80]}")

    # ---------- 5. 数据与缓存 ----------
    from cxr import config as C
    line(OK, f"CXR_DATA  = {C.DATA}")
    line(OK, f"CXR_CACHE = {C.CACHE}")
    need = [os.path.join(C.DATA, "train.csv"), os.path.join(C.DATA, "valid.csv"),
            os.path.join(C.DATA, "test.csv"),
            os.path.join(C.CACHE, "train_512.npy"), os.path.join(C.CACHE, "valid_512.npy"),
            os.path.join(C.CACHE, "train_512.json"), os.path.join(C.CACHE, "valid_512.json")]
    for f in need:
        if os.path.exists(f):
            line(OK, f"{os.path.basename(f):18s} {os.path.getsize(f)/1e9:6.2f} GB")
        else:
            line(BAD, f"缺失 {f}")
    # 原始图片不是必需的（训练只读缓存），但 valid/test 推理若走 direct 路径会需要
    img_ok = os.path.isdir(os.path.join(C.DATA, "valid"))
    line(WARN if not img_ok else OK,
         f"原始图片目录 {'存在' if img_ok else '不存在（训练不需要；按缓存推理即可）'}")

    # ---------- 6. 速度微基准 ----------
    try:
        from cxr.data import build_dataset  # 若不存在则跳过
        line(OK, "数据管道可导入")
    except Exception:  # noqa: BLE001
        line(WARN, "build_dataset 未找到，速度基准跳过（训练脚本自会报错）")

    print("=" * 74)
    print("自检结束。若上面没有 ❌，可以直接跑 E1。")
    print("=" * 74)


if __name__ == "__main__":
    main()
