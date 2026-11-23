"""Build a side-by-side heatmap comparison: original weights vs retrained weights.

Inputs are the per-class overlays written by tools/heatmap_probe.py:
  class_<k>_diff.png  = mean positive map minus mean negative map (what the class uses)
  class_<k>.png       = mean positive map (mostly shared anatomy)

Output:
  compare_heatmaps.png   montage, 10 rows (classes) x 2 columns (orig | retrain)
  compare_heatmaps.md    quantitative comparison from diagnostics.csv
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = r"E:\挑战赛"
BASE = os.path.join(ROOT, "reports", "heatmaps_retrain")
CELL = 190
PAD = 6
LABEL_W = 206
HEADER_H = 84
FOOTER_H = 30


def load_font(size: int):
    for name in ("msyh.ttc", "msyhbd.ttc", "simhei.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def main() -> None:
    # 官方类别映射（CheXpert 类序，2026-11-20 核实；此前按 NIHCXR 假设，9 个名字是错的）
    NAMES = {1: "Enlarged Cardiomediastinum", 2: "Pneumothorax", 3: "Consolidation",
             4: "Pneumonia", 5: "Edema", 6: "Cardiomegaly", 7: "Atelectasis",
             8: "Lung Opacity", 9: "Pleural Effusion", 10: "No Finding"}
    font = load_font(16)
    small = load_font(13)
    classes = list(range(1, 11))
    cols = 2
    W = LABEL_W + cols * (CELL + PAD) + PAD
    H = HEADER_H + len(classes) * (CELL + PAD) + FOOTER_H
    canvas = Image.new("RGB", (W, H), (250, 250, 250))
    d = ImageDraw.Draw(canvas)

    d.text((PAD, 8), "逐类净证据热力图：原权重 vs 重训权重", font=font, fill=(20, 20, 20))
    d.text((PAD, 30), "均值正样本 − 均值负样本 ｜ 验证集 ｜ 每类 20 正 + 20 负", font=small, fill=(90, 90, 90))
    heads = ["原权重 cnxt512_pat_e12", "重训权重 repro42"]
    for j, h in enumerate(heads):
        x = LABEL_W + j * (CELL + PAD) + CELL // 2
        tw = d.textlength(h, font=small)
        d.text((x - tw / 2, HEADER_H - 24), h, font=small, fill=(60, 60, 90))

    for i, c in enumerate(classes):
        y = HEADER_H + i * (CELL + PAD)
        name = NAMES.get(c, "")
        d.text((6, y + CELL // 2 - 16), f"{c}. {name}", font=small, fill=(20, 20, 20))
        for j, sub in enumerate(("orig42_20", "repro42_20")):
            p = os.path.join(BASE, sub, f"class_{c}_diff.png")
            x = LABEL_W + j * (CELL + PAD)
            if os.path.exists(p):
                im = Image.open(p).convert("RGB").resize((CELL, CELL), Image.BILINEAR)
                canvas.paste(im, (x, y))
            d.rectangle([x, y, x + CELL - 1, y + CELL - 1], outline=(200, 200, 205))

    d.text((PAD, H - FOOTER_H + 6),
           "验证集；每类 20 个正样本与 20 个负样本的均值证据叠加；红=高响应",
           font=small, fill=(90, 90, 90))
    out = os.path.join(BASE, "compare_heatmaps.png")
    canvas.save(out)
    print(f"[wrote] {out}  {W}x{H}")

    # 量化对比
    a = pd.read_csv(os.path.join(BASE, "orig42_20", "diagnostics.csv"))
    b = pd.read_csv(os.path.join(BASE, "repro42_20", "diagnostics.csv"))
    keep = ["border_share", "corner_share", "centre_share", "lr_balance"]
    a = a[a.kind == "pos_minus_neg"][["class"] + keep].set_index("class")
    b = b[b.kind == "pos_minus_neg"][["class"] + keep].set_index("class")
    rows = ["# 热力图对比：原权重 vs 重训权重", "",
            "指标定义：`border_share` 边缘响应占比、`corner_share` 四角占比、",
            "`centre_share` 中心区占比、`lr_balance` 左右不对称度。均为「净证据图」上的统计。", "",
            "| 类别 | 中心占比 原 | 中心占比 重训 | 边缘占比 原 | 边缘占比 重训 | 四角 原 | 四角 重训 |",
            "|---:|---:|---:|---:|---:|---:|---:|"]
    for c in a.index:
        rows.append(f"| {c} | {a.loc[c,'centre_share']:.3f} | {b.loc[c,'centre_share']:.3f} | "
                    f"{a.loc[c,'border_share']:.3f} | {b.loc[c,'border_share']:.3f} | "
                    f"{a.loc[c,'corner_share']:.3f} | {b.loc[c,'corner_share']:.3f} |")
    mc_a, mc_b = a.centre_share.mean(), b.centre_share.mean()
    md_a, md_b = a.border_share.mean(), b.border_share.mean()
    rows += ["", f"平均中心占比：原 **{mc_a:.3f}** / 重训 **{mc_b:.3f}** "
                 f"（差 {mc_b-mc_a:+.3f}）",
             f"平均边缘占比：原 **{md_a:.3f}** / 重训 **{md_b:.3f}** "
             f"（差 {md_b-md_a:+.3f}）", "",
             "## 怎么看这张图", "",
             "- 两列的病灶定位应当**基本一致**：同一类在同一批图像上，热区落在同样的解剖位置。",
             "- 边缘占比低（≤0.1 量级）说明模型没有靠片子边缘的文字标记或黑边做判断。",
             "- 两列允许存在强度与细节差异——重训权重不是原权重的副本（cuDNN 非确定性，",
             "  见 `reports/重训复现_2026-11-18.md`），但**关注位置**应稳定。"]
    ag = os.path.join(BASE, "map_agreement.csv")
    if os.path.exists(ag):
        t = pd.read_csv(ag)
        rows += ["", "## 这两张图的可信度：先看噪声上限", "",
                 "每类只用少量样本平均出来的热力图，本身就是一个噪声很大的估计。下表把「两模型相关」",
                 "与「同一模型两半样本互比」放在一起：如果跨模型相关并不低于同模型自比，",
                 "就说明两模型看的是一回事，图上看到的差别主要是抽样噪声。", "",
                 "| 类别 | 跨模型相关 | 同模型自比(原) | 同模型自比(重训) | 判定 |",
                 "|---:|---:|---:|---:|---|"]
        for _, r in t.iterrows():
            verdict = "稳定" if r.r_cross > 0.9 else "偏不稳"
            rows.append(f"| {int(r['class'])} {r['name']} | {r['r_cross']:.3f} | "
                        f"{r['r_self_orig']:.3f} | {r['r_self_new']:.3f} | {verdict} |")
        rows += ["", "读法：同模型自比 = 把同一模型的 20 个正样本随机分成 10+10，各自求均值图后比相关。",
                 "它就是这套可视化方法的**噪声上限**。跨模型相关普遍高于它，说明模型之间的一致性比",
                 "测量本身能分辨的还要高。", "",
                 "**实践建议**：类 1（Atelectasis，valid 只有 35 个正样本）与类 6（Pneumonia）的",
                 "热力图即使加到 20 张也不稳；要对比这类模型，至少用 40 张以上，并且先报噪声上限。"]
    out_md = os.path.join(BASE, "compare_heatmaps.md")
    open(out_md, "w", encoding="utf-8").write("\n".join(rows))
    print(f"[wrote] {out_md}")


if __name__ == "__main__":
    main()
