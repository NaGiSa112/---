"""从 runs/ 的真实日志恢复实验登记与 GPU 分钟账单（审阅 §9 第 2a 步）。

规则：**只读 history.json / config.json 里记录的实际耗时与配置**，不用估算；
缺失就写「未记录」。同时列出教师/学生/供体资产清单。

    python tools/build_experiment_ledger.py
"""
from __future__ import annotations

import json
import os
import sys

ROOT = r"E:\挑战赛"
RUNS = os.path.join(ROOT, "runs")
OUT = os.path.join(ROOT, "04_新一轮", "实验登记表.md")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def load_json(p: str):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return None


def summarise(run_dir: str) -> dict | None:
    h = os.path.join(run_dir, "history.json")
    if not os.path.exists(h):
        return None
    d = load_json(h)
    hist = d.get("history", d) if isinstance(d, dict) else d
    if not isinstance(hist, list) or not hist:
        return None
    cfg = {}
    c = os.path.join(run_dir, "config.json")
    if os.path.exists(c):
        cfg = load_json(c) or {}
    secs = [e.get("sec") for e in hist if isinstance(e.get("sec"), (int, float))]
    best = max(hist, key=lambda e: e.get("val_auc", -1))
    return {
        "run": os.path.basename(run_dir),
        "epochs": len(hist),
        "sec_total": float(sum(secs)) if secs else None,
        "sec_missing": len(hist) - len(secs),
        "best_epoch": best.get("epoch"),
        "best_val": best.get("val_auc"),
        "last_val": hist[-1].get("val_auc"),
        "arch": cfg.get("arch"),
        "img_size": cfg.get("img_size"),
        "batch": cfg.get("batch_size"),
        "seed": cfg.get("seed"),
        "kd": cfg.get("distill_weight"),
        "patient_smooth": cfg.get("patient_smooth"),
        "ps_c1": cfg.get("patient_smooth_c1"),
        "lse": cfg.get("lse_alpha"),
        "extra_channel": cfg.get("extra_channel"),
        "blend48": cfg.get("label_blend_48"),
        "init_weights": cfg.get("init_weights"),
        "fold": cfg.get("valid_fold"),
        "train_folds": cfg.get("train_folds"),
    }


def main() -> None:
    rows = []
    for name in sorted(os.listdir(RUNS)):
        d = os.path.join(RUNS, name)
        if not os.path.isdir(d):
            continue
        s = summarise(d)
        if s:
            rows.append(s)

    total_sec = sum(r["sec_total"] or 0 for r in rows)
    L = ["# 实验登记表（2026-11-20 生成）", "",
         "> 全部数字来自 `runs/*/history.json` 的 `sec` 字段与 `config.json`，**不是估算**。",
         f"> 扫描到 {len(rows)} 个带 history 的运行目录，合计 **{total_sec/3600:.2f} GPU 小时**。", "",
         "## 一、全部运行（按耗时降序）", "",
         "| run | 轮 | 实际耗时(分) | 最优轮 | 最优 val | 末轮 val | arch | 输入 | batch | seed | KD | 患者平滑 | LSE | 其它 | fold |",
         "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---:|"]

    for r in sorted(rows, key=lambda x: -(x["sec_total"] or 0)):
        mins = f"{r['sec_total']/60:.1f}" if r["sec_total"] else "未记录"
        extra = []
        if r["extra_channel"]:
            extra.append(f"通道={r['extra_channel']}")
        if r["blend48"]:
            extra.append(f"blend48={r['blend48']}")
        if r["ps_c1"] is not None and r["ps_c1"] >= 0:
            extra.append(f"类1平滑={r['ps_c1']}")
        if r["init_weights"]:
            extra.append("22K初始化")
        L.append("| `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r["run"], r["epochs"], mins, r["best_epoch"],
            f"{r['best_val']:.4f}" if isinstance(r["best_val"], float) else "—",
            f"{r['last_val']:.4f}" if isinstance(r["last_val"], float) else "—",
            r["arch"] or "—", r["img_size"] or "—", r["batch"] or "—", r["seed"] or "—",
            r["kd"] or 0, r["patient_smooth"] or 0, r["lse"] or 0,
            "；".join(extra) or "—", r["fold"] if r["fold"] is not None else "—"))

    # ---- OOF 五折与候选累计 ----
    L += ["", "## 二、按用途分组的小计", ""]
    groups = {
        "OOF 基线（oof_f0..4）": [r for r in rows if r["run"].startswith("oof_f")],
        "受控重复 A/B（repA42/repB42）": [r for r in rows if r["run"].startswith("rep")],
        "B1 差分通道（asym_f0..4）": [r for r in rows if r["run"].startswith("asym_f")],
        "B2 软标签（blend48_f0..4）": [r for r in rows if r["run"].startswith("blend48_f")],
        "22K 初始化（init22k_f0..1）": [r for r in rows if r["run"].startswith("init22k_f")],
        "类 1 平滑（pat_c1_f0..1）": [r for r in rows if r["run"].startswith("pat_c1_f")],
        "Swin-T（swin_b4）": [r for r in rows if r["run"].startswith("swin")],
    }
    L += ["| 分组 | 运行数 | 实际耗时 |", "|---|---:|---:|"]
    for g, items in groups.items():
        t = sum(r["sec_total"] or 0 for r in items)
        L.append(f"| {g} | {len(items)} | {t/3600:.2f} 小时 |")

    # ---- 资产清单 ----
    L += ["", "## 三、权重与资产清单", "",
          "| 资产 | 路径 | 大小 | 备注 |", "|---|---|---:|---|"]
    assets = [
        ("Tiny 种子42（交付）", "runs/cnxt512_pat_e12/best.pt"),
        ("Tiny 种子43（交付）", "runs/cnxt512_pat_s43_e12/best.pt"),
        ("Tiny 种子44（交付）", "releases/final_day_seed44_2026-11-13/model/tiny_seed44_last12.pt"),
        ("B3 供体（交付）", "releases/final_day_seed44_2026-11-13/model/donor_v21.pt"),
        ("Swin-T（B4 候选）", "runs/swin_b4/best.pt"),
        ("22K 转换权重", "runs/init22k/tiny22k_adapted.pt"),
        ("教师软标签（全 train）", "reports/teacher_soft_full_a.csv"),
    ]
    for name, rel in assets:
        p = os.path.join(ROOT, rel.replace("/", os.sep))
        sz = f"{os.path.getsize(p)/1e6:.2f} MB" if os.path.exists(p) else "缺失"
        L.append(f"| {name} | `{rel}` | {sz} | |")

    # ---- 教师/学生谱系 ----
    L += ["", "## 四、KD 谱系与「7 小时」的账单口径", "",
          "- 教师：交付版的 3 个 Tiny 中取两只做软标签教师，`reports/teacher_soft_full_a.csv` 是**全 train** 生成的。",
          "  ⇒ 因此 OOF 阶段一律 no-KD，否则教师见过外层留出折（泄漏）。",
          "- 若要重建「干净部署 OOF」：每折需要 2 个教师 + 3 个学生 + B3 供体，逐折隔离。",
          "  按本表实测单折学生耗时（见上表 `oof_f*` 行）可推算，但**教师训练没有历史记录**，",
          "  所以「7 小时」是**假设值而非实测**，审阅指出这一点是正确的。",
          "", "| 组成部分 | 是否有实测记录 |", "|---|---|",
          "| 学生单折（no-KD） | ✅ 有（见上表） |",
          "| 学生单折（含 KD） | ❌ 无独立记录 |",
          "| 教师单折 | ❌ 无记录 |",
          "| B3 供体单折 | ❌ 无记录（协作方训练） |",
          "| 软标签生成 | ❌ 无计时 |"]

    open(OUT, "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L[:30]))
    print(f"\n  [wrote] {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
