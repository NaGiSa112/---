"""给受「类别映射更正」影响的报告加横幅（2026-11-20）。

只加横幅，不改正文 —— 保留当时的判断记录。
"""
from __future__ import annotations

import os
import sys

ROOT = r"E:\挑战赛"

BANNER = """⚠️ **2026-11-20 类别映射更正**：官方 10 类用的是 **CheXpert 类序**，我们此前按 NIHCXR 类序假设，
**9 个名字是错的**（仅 10 号相同）。官方：1 纵隔心影增大、2 气胸、3 肺部实变、4 肺炎、5 肺水肿、
6 心影扩大、7 肺不张、8 肺部阴影、9 胸腔积液、10 无异常发现。
本文中的**病名解释与机理推断需按此重读**；数字类上的实验结论不变（全流程按数字 id 运作）。
详见 `reports/类别映射更正_2026-11-20.md`。"""

TARGETS = {
    "reports/B1_差分通道_负结果_2026-11-19.md":
        "「左右对称差分针对类 1（当时误认为肺不张）」的机理理由撤回：类 1 实际是**纵隔心影增大**。"
        "实验结论（五折宏 Δ≈0）仍成立。",
    "reports/B2_软标签_结果_2026-11-20.md":
        "「P(8|4)=0.529 说明标注边界模糊」的解释撤回：类 4 是**肺炎**、类 8 是**肺部阴影**，"
        "共现是因为同时为真而非标错 —— 这恰好解释了软标签为什么失败。",
    "reports/候选结果_2026-11-20.md":
        "22K 的逐类结论按数字类仍成立；「类 2 掉 0.0191」指的是**气胸**，不是心影。",
    "reports/逐类收缩系数_2026-11-19.md":
        "「心影大小是患者固有属性，所以类 2 用 0.50 更稳」的机制解释撤回：类 2 是**气胸**。"
        "数值结果不变，机制待查。",
    "reports/弱类排序_2026-11-19.md":
        "表中类名为旧假设。真实弱类：1 纵隔心影增大（0.695）、4 肺炎（0.733）、8 肺部阴影（0.754）。",
    "reports/heatmaps_retrain/展示口径变更_2026-11-19.md":
        "文中类名为旧假设，须按官方 CheXpert 类序重读。",
    "04_新一轮/阶段进展评审包_2026-11-20.md":
        "本文的弱类解释按旧病名写成；数字全部不变，病名按更正报告重读。",
    "04_新一轮/当前问题清单_2026-11-20.md":
        "问题 C1（类别→病名未确认）已升级并部分解决：官方映射已核实归档，9 个旧名作废。",
    "04_新一轮/国赛提分方案_2026-11-19.md":
        "文中病名为旧假设；官方映射见 `reports/类别映射更正_2026-11-20.md`。",
}


def main() -> None:
    for rel, extra in TARGETS.items():
        p = os.path.join(ROOT, rel.replace("/", os.sep))
        if not os.path.exists(p):
            print(f"  缺失: {rel}")
            continue
        s = open(p, encoding="utf-8").read()
        if "2026-11-20 类别映射更正" in s:
            print(f"  已有横幅: {rel}")
            continue
        lines = s.split("\n")
        out, done = [], False
        for l in lines:
            out.append(l)
            if not done and l.startswith("# "):
                out += ["", "> " + BANNER.replace("\n", "\n> "), "",
                        "> 本条专属：" + extra]
                done = True
        if not done:
            out = ["> " + BANNER.replace("\n", "\n> "), ""] + out
        open(p, "w", encoding="utf-8").write("\n".join(out))
        print(f"  已加横幅: {rel}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    main()
