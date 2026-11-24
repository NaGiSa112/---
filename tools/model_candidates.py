"""候选主干的硬指标（不下载权重，只读 timm 的模型注册信息）。

输出：参数量、默认输入、timm 记录的 ImageNet top-1、许可证、以及"相对 Tiny 的算力倍数"估算。

    python tools/model_candidates.py
"""
from __future__ import annotations

import json
import os
import sys

ROOT = r"E:\挑战赛"
OUT = os.path.join(ROOT, "reports", "model_candidates_2026-11-24.md")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# (魔搭 repo, timm 名, 许可证, 预训练数据域, 备注)
CANDS = [
    ("timm/convnextv2_tiny.fcmae_ft_in22k_in1k", "convnextv2_tiny.fcmae_ft_in22k_in1k",
     "cc-by-nc-4.0", "ImageNet-22K + FCMAE 自监督", "非商用许可，需谨慎"),
    ("timm/convnextv2_base.fcmae_ft_in22k_in1k", "convnextv2_base.fcmae_ft_in22k_in1k",
     "cc-by-nc-4.0", "ImageNet-22K + FCMAE", "非商用许可；容量 ~3.8x Tiny"),
    ("timm/caformer_s18.sail_in22k_ft_in1k", "caformer_s18.sail_in22k_ft_in1k",
     "apache-2.0", "ImageNet-22K（SAIL 蒸馏预训练）", "MetaFormer + 深度可分离卷积"),
    ("timm/maxvit_tiny_tf_224.in1k", "maxvit_tiny_tf_224.in1k",
     "apache-2.0", "ImageNet-1K", "多轴注意力，归纳偏置不同"),
    ("timm/maxvit_small_tf_224.in1k", "maxvit_small_tf_224.in1k",
     "apache-2.0", "ImageNet-1K", "同上，容量更大"),
    ("timm/eva02_base_patch14_448.mim_in22k_ft_in22k_in1k",
     "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k",
     "mit", "ImageNet-22K + MIM", "强表征，但 448 输入很贵"),
    ("timm/swinv2_tiny_window16_256.ms_in1k", "swinv2_tiny_window16_256.ms_in1k",
     "mit", "ImageNet-1K（ImageNet-21K 蒸馏）", "已测过 Swin-T v1 无增益"),
    ("timm/repvit_m2_3.dist_300e_in1k", "repvit_m2_3.dist_300e_in1k",
     "apache-2.0", "ImageNet-1K（蒸馏）", "移动端，很快"),
    ("timm/mobilenetv4_conv_medium.e500_r224_in1k", "mobilenetv4_conv_medium.e500_r224_in1k",
     "apache-2.0", "ImageNet-1K", "很轻，容量可能不足"),
    ("timm/efficientvit_b1.r288_in1k", "efficientvit_b1.r288_in1k",
     "apache-2.0", "ImageNet-1K", "288 输入，效率高"),
    ("timm/coatnet_1_rw_224.sw_in1k", "coatnet_1_rw_224.sw_in1k",
     "apache-2.0", "ImageNet-1K（弱监督）", "卷积+注意力混合"),
    ("(baseline) 我们现用", "convnext_tiny.fb_in22k_ft_in1k",
     "apache-2.0", "ImageNet-22K", "当前基座"),
]


def main() -> None:
    import timm
    rows = []
    for repo, name, lic, data, note in CANDS:
        try:
            m = timm.create_model(name, pretrained=False, num_classes=10)
            params = sum(p.numel() for p in m.parameters()) / 1e6
            cfg = m.pretrained_cfg
            top1 = cfg.get("top1", "—")
            img = cfg.get("input_size", (3, 224, 224))[-1]
            rows.append(dict(repo=repo, timm=name, license=lic, data=data, note=note,
                             params_m=round(params, 1), img=int(img), top1=top1))
            print(f"  {name[:46]:46s} {params:6.1f}M  {img:3d}px  top1={top1}")
        except Exception as e:  # noqa: BLE001
            print(f"  {name[:46]:46s} 失败: {type(e).__name__}: {str(e)[:50]}")

    base = next((r for r in rows if r["timm"].startswith("convnext_tiny")), None)
    L = ["# 候选主干硬指标（2026-11-24，来自 timm 注册信息）", "",
         "> 约束：通用图像预训练（禁医学权重）；加载权重 ≤500 MB；单张 ≤100 ms（现状 377 MB / 51 ms）。",
         "", "| 候选 | 参数量 | 输入 | ImageNet top-1 | 许可证 | 预训练数据 | 备注 |",
         "|---|---:|---:|---:|---|---|---|"]
    for r in rows:
        L.append(f"| `{r['timm']}` | {r['params_m']}M | {r['img']} | {r['top1']} | "
                 f"{r['license']} | {r['data']} | {r['note']} |")
    if base:
        L += ["", f"基线 {base['timm']}：{base['params_m']}M、{base['img']}px、"
                  f"top1={base['top1']}。上表参数量可直接对比容量倍数。"]
    L += ["", "## 挑选用法", "",
          "1. **许可证先过筛**：`cc-by-nc-4.0` 属非商用，比赛场景虽可能被允许，但交付会带来不确定性；",
          "   优先 `apache-2.0` / `mit`。",
          "2. **容量倍数看参数**：我们已实测 ConvNeXt-Small/Base、Swin-T 均无增益甚至更差；",
          "   再上更大的模型，除非换预训练范式，否则大概率重复负结果。",
          "3. **唯一有新信息量的是\"不同的归纳偏置或预训练范式\"**：",
          "   MaxViT（多轴注意力）、CAFormer（MetaFormer + 深度可分离）、EVA-02（MIM 预训练）。",
          "4. **部署代价**：EVA-02-B/448 与 MaxViT-S 在 100 ms 预算下几乎只能单独使用，",
          "   不能作为第 5 个成员（现包已用 377 MB / 51 ms）。"]
    open(OUT, "w", encoding="utf-8").write("\n".join(L))
    json.dump(rows, open(os.path.join(ROOT, "reports", "model_candidates.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n  [wrote] {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
