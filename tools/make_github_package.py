"""组装一个可放进 GitHub 的轻量仓库包。

原则
----
* **只放代码、文档、小结果**：不含权重（*.pt）、数据（cxr_first/）、缓存（*.npy）、
  运行目录（runs/）、发布包（releases/、submissions/）、依赖（.pylibs/）。
* 所有文件计算 SHA-256，写进 MANIFEST.json 与 MANIFEST.md。
* 目录结构按"能直接 git init & push"设计：README 在根、.gitignore 到位。

    python tools/make_github_package.py
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys

ROOT = r"E:\挑战赛"
OUT = os.path.join(ROOT, "05_github_exports", "cxr-2026-guosai")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# (源相对路径, 目标相对路径)
FILES = [
    # ---- 文档 ----
    ("04_新一轮/进度总结_给队友_2026-11-21.md", "docs/00_进度总结.md"),
    ("04_新一轮/国赛30%目标策略_2026-11-21.md", "docs/01_国赛策略_前30%.md"),
    ("04_新一轮/当前问题清单_2026-11-20.md", "docs/02_问题清单.md"),
    ("04_新一轮/历史选型记录_2026-11-20.md", "docs/03_历史选型记录.md"),
    ("04_新一轮/实验登记表.md", "docs/04_实验登记表.md"),
    ("04_新一轮/阶段进展评审包_2026-11-20.md", "docs/05_评审包.md"),
    ("reports/类别映射更正_2026-11-20.md", "docs/10_类别映射更正.md"),
    ("reports/基线核验_2026-11-20.md", "docs/11_基线核验.md"),
    ("reports/类2回放审计_2026-11-20.md", "docs/12_类2回放审计.md"),
    ("reports/患者上下文审计_2026-11-20.md", "docs/13_患者上下文审计.md"),
    ("reports/名次密度分析_2026-11-21.md", "docs/14_名次密度分析.md"),
    ("reports/弱类排序_2026-11-19.md", "docs/20_弱类排序.md"),
    ("reports/逐类收缩系数_2026-11-19.md", "docs/21_逐类收缩.md"),
    ("reports/B1_差分通道_负结果_2026-11-19.md", "docs/22_B1_差分通道.md"),
    ("reports/B2_软标签_结果_2026-11-20.md", "docs/23_B2_软标签.md"),
    ("reports/B4_Swin成员_负结果_2026-11-19.md", "docs/24_B4_Swin.md"),
    ("reports/候选结果_2026-11-20.md", "docs/25_候选_22K与类1平滑.md"),
    ("reports/折bagging_2026-11-21.md", "docs/26_折bagging.md"),
    ("reports/DINOv2探针_负结果_2026-11-22.md", "docs/38_DINOv2探针.md"),
    ("reports/逐位复现记录_2026-11-21.md", "docs/39_逐位复现记录.md"),
    ("reports/heatmaps_retrain/展示口径变更_2026-11-19.md", "docs/27_热力图口径.md"),
    ("reports/审阅更正_2026-11-18.md", "docs/28_上一轮审阅更正.md"),
    ("reports/今晚问题报告_2026-11-18.md", "docs/29_11-18问题报告.md"),
    ("reports/重训复现_2026-11-18.md", "docs/30_重训复现.md"),
    ("reports/数据画像_2026-11-18.md", "docs/31_数据画像.md"),
    # ---- 被文档引用、但早期遗漏的文件（避免索引里的路径失效）----
    ("04_新一轮/00_规则/patient_folds_5f.csv", "data/patient_folds_5f.csv"),
    ("reports/弱类整改方案_2026-11-18.md", "docs/32_弱类整改方案.md"),
    ("04_新一轮/方案评审包_2026-11-18.md", "docs/33_方案评审包_11-18.md"),
    ("04_新一轮/国赛方案_2026-12-01.md", "docs/34_国赛方案_11-18版.md"),
    ("04_新一轮/04_文档/方案可行性审阅_2026-11-18.md", "docs/35_外部审阅_11-18.md"),
    ("reports/SWIN_CROSS_ARCH_2026-11-07.md", "docs/36_Swin跨架构_11-07.md"),
    ("reports/ENS3_2026-11-07.md", "docs/37_三成员集成_11-07.md"),
    ("reports/heatmaps_retrain/map_agreement.csv", "data/map_agreement.csv"),
    ("reports/teacher_soft_full_a.csv", "data/teacher_soft_full_a.csv"),
    # ---- 异构骨干实验（2026-11-24）----
    ("reports/CAFormer实验报告_v2_2026-11-24.md", "docs/40_CAFormer实验_v2.md"),
    ("reports/CAFormer实验报告_2026-11-24.md", "docs/41_CAFormer实验_v1作废.md"),
    ("04_新一轮/扩充成员方案_2026-11-24.md", "docs/42_扩充成员方案.md"),
    ("04_新一轮/多模型异构集成方案_2026-11-24.md", "docs/43_异构集成方案.md"),
    ("04_新一轮/GPU预算重排_19小时天_2026-11-24.md", "docs/44_GPU预算重排.md"),
    ("04_新一轮/冲分方案_2026-11-23.md", "docs/45_冲分方案.md"),
    ("04_新一轮/国赛7天提交阶梯_2026-11-23.md", "docs/46_国赛提交阶梯.md"),
    ("reports/成员强度_2026-11-24.md", "docs/47_成员强度.md"),
    ("reports/魔搭候选主干_2026-11-24.md", "docs/48_魔搭候选主干.md"),
    ("reports/CAFormer与Transformer_区别_2026-11-24.md", "docs/49_CAFormer与Transformer.md"),
    ("reports/后处理网格_2026-11-23.md", "docs/50_后处理网格.md"),
    ("reports/E1E2v2_判读_caformer.md", "results/E1E2v2_caformer.md"),
    ("reports/E1E2v2_判读_caformer.json", "results/E1E2v2_caformer.json"),
    ("reports/member_strength.json", "results/member_strength.json"),
    ("reports/arch_latency_2026-11-24.json", "results/arch_latency.json"),
    ("reports/model_candidates.json", "results/model_candidates.json"),
    ("reports/postproc_grid_2026-11-23.json", "results/postproc_grid.json"),
    ("tools/e1_score_v2.py", "tools/e1_score_v2.py"),
    ("tools/e1_score.py", "tools/e1_score.py"),
    ("tools/make_probs_from_ckpt.py", "tools/make_probs_from_ckpt.py"),
    ("tools/member_strength.py", "tools/member_strength.py"),
    ("tools/arch_latency.py", "tools/arch_latency.py"),
    ("tools/model_candidates.py", "tools/model_candidates.py"),
    ("tools/e0_adapter_check.py", "tools/e0_adapter_check.py"),
    ("tools/cloud_env_check.py", "tools/cloud_env_check.py"),
    ("tools/e1_local_queue.py", "tools/e1_local_queue.py"),
    ("tools/bank_variants.py", "tools/bank_variants.py"),
    ("tools/check_submission.py", "tools/check_submission.py"),
    ("tools/postproc_grid.py", "tools/postproc_grid.py"),
    ("reports/E1E2_判读_caformer_2026-11-24.md", "results/E1E2v1_caformer.md"),
    ("reports/E1E2_判读_caformer_2026-11-24.json", "results/E1E2v1_caformer.json"),
    # ---- 计划与演练（2026-11-25）----
    ("04_新一轮/后续计划总览_2026-11-25.md", "docs/00b_后续计划总览.md"),
    ("04_新一轮/GPU240小时编排_2026-11-25.md", "docs/44b_GPU240小时编排.md"),
    ("reports/交付演练_2026-11-25.md", "docs/51_交付演练.md"),
    ("reports/Transformer轴结论_2026-11-24.md", "docs/52_Transformer轴结论.md"),
    ("reports/E1E2v2_判读_swin_t.md", "results/E1E2v2_swin_t.md"),
    ("reports/E1E2v2_判读_swin_t.json", "results/E1E2v2_swin_t.json"),
    # ---- 小数据 ----
    ("04_新一轮/00_规则/patient_folds_5f.json", "data/patient_folds_5f.json"),
    ("04_新一轮/初赛前100名_2026-11-21.csv", "data/初赛前100名.csv"),
    ("reports/弱类排序_2026-11-19.csv", "data/弱类排序.csv"),
    ("reports/数据画像_2026-11-18.csv", "data/数据画像.csv"),
    ("reports/baseline_audit_2026-11-20.json", "results/baseline_audit.json"),
    ("reports/class2_audit_2026-11-20.json", "results/class2_audit.json"),
    ("reports/context_audit_2026-11-20.json", "results/context_audit.json"),
    ("reports/per_class_shrink.json", "results/per_class_shrink.json"),
    ("reports/dinov2_probe_2026-11-21.json", "results/dinov2_probe.json"),
    ("reports/leaderboard_2026-11-21.json", "results/leaderboard.json"),
    ("reports/resolution_test.json", "results/resolution_test.json"),
    ("reports/outside_body/band_evidence_by_class.csv", "data/边框带证据.csv"),
    ("reports/outside_body/border_ablation.json", "results/border_ablation.json"),
]

TOOL_DIRS = {
    "tools/": [
        "study_metric.py", "verify_predictions.py", "compare_folds.py",
        "audit_baseline.py", "audit_class2_full.py", "audit_class2_gain.py",
        "audit_patient_context.py", "rank_weak_classes.py", "fit_per_class_shrink.py",
        "analyse_leaderboard.py", "eval_fold_bagging.py", "dinov2_probe.py",
        "dinov2_features.py", "convert_convnext22k.py", "run_folds.py",
        "chain_run.py", "make_patient_folds.py", "heatmap_probe.py",
        "show_class_samples.py", "make_heatmap_compare.py", "evidence_outside_body.py",
        "ablate_border.py", "resolution_test.py", "data_profile.py",
        "diagnose_weak_classes.py", "why_class56_probe.py", "overnight_queue.py",
        "overnight_report.py", "add_label_banners.py", "build_experiment_ledger.py",
        "patient_structure_impact.py", "make_github_package.py",
    ],
}

SRC_FILES = [
    "src/cxr/__init__.py", "src/cxr/config.py", "src/cxr/models.py", "src/cxr/data.py",
    "src/cxr/cache.py", "src/cxr/infer.py", "src/cxr/train.py", "src/cxr/metrics.py",
    "src/cxr/deploy.py", "src/predict.py", "src/evaluate.py", "src/build_cache.py",
]

GITIGNORE = """# 大文件与运行产物 —— 不进仓库
*.pt
*.pth
*.safetensors
*.npy
*.npz
*.zip
runs/
releases/
submissions/
cache/
cxr_first/
.pylibs/
.torch_home/
05_github_exports/
__pycache__/
*.pyc
.ipynb_checkpoints/
.tmp/
"""


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def copy(rel_src: str, rel_dst: str, rows: list) -> None:
    s = os.path.join(ROOT, rel_src.replace("/", os.sep))
    d = os.path.join(OUT, rel_dst.replace("/", os.sep))
    if not os.path.exists(s):
        print(f"  跳过（缺失）: {rel_src}")
        return
    os.makedirs(os.path.dirname(d), exist_ok=True)
    shutil.copy2(s, d)
    rows.append({"path": rel_dst, "bytes": os.path.getsize(d), "sha256": sha256(d),
                 "src": rel_src})


def main() -> None:
    if os.path.isdir(OUT):
        shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT, exist_ok=True)
    rows: list[dict] = []

    for rel_src, rel_dst in FILES:
        copy(rel_src, rel_dst, rows)
    for _, names in TOOL_DIRS.items():
        for n in names:
            copy(f"tools/{n}", f"tools/{n}", rows)
    for rel in SRC_FILES:
        copy(rel, rel.replace("src/", "src/"), rows)

    open(os.path.join(OUT, ".gitignore"), "w", encoding="utf-8").write(GITIGNORE)

    total = sum(r["bytes"] for r in rows)
    by_dir: dict[str, list] = {}
    for r in rows:
        by_dir.setdefault(r["path"].split("/")[0], []).append(r)

    man_md = ["# MANIFEST", "",
              f"共 **{len(rows)}** 个文件，合计 **{total/1024:.1f} KB**。", "",
              "| 目录 | 文件数 | 大小 |", "|---|---:|---:|"]
    for d, items in sorted(by_dir.items()):
        man_md.append(f"| `{d}/` | {len(items)} | {sum(i['bytes'] for i in items)/1024:.1f} KB |")
    man_md += ["", "## 逐文件 SHA-256", "", "| 文件 | 字节 | SHA-256 |", "|---|---:|---|"]
    for r in sorted(rows, key=lambda x: x["path"]):
        man_md.append(f"| `{r['path']}` | {r['bytes']} | `{r['sha256'][:32]}…` |")
    open(os.path.join(OUT, "MANIFEST.md"), "w", encoding="utf-8").write("\n".join(man_md))
    json.dump(rows, open(os.path.join(OUT, "MANIFEST.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print(f"  共 {len(rows)} 个文件，{total/1024:.1f} KB")
    for d, items in sorted(by_dir.items()):
        print(f"   {d+"/":16s} {len(items):3d} 文件  {sum(i['bytes'] for i in items)/1024:8.1f} KB")
    print(f"  [out] {OUT}")


if __name__ == "__main__":
    main()
