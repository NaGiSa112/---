"""Overnight queue (2026-11-18 23:55 -> 2026-11-19 08:00).

执行顺序刻意按"即使被 8:00 打断、已完成的阶段仍然可用"来排：

  S0  前置自检（折表、指标工具、缓存）
  S1-5 五折干净 OOF（no-KD、固定 last12、冻结折表）—— 最耗时的核心产出
  S6  汇总 OOF：逐类 study 级 AUC + 与历史记录对照
  S7  受控重复 A（完整部署配方、seed 42、KD）
  S8  受控重复 B（与 A 完全相同的参数）
  S9  跑两个重复在官方 valid 上的检查级 AUC → 得到"同一配方跑两次"的波动实测值
  S10 写总结 markdown + status.json

每个阶段完成即写 runs/overnight/status.json，并把日志追加到 runs/overnight/queue.log。
重复运行时已完成的阶段会自动跳过（按输出存在判断）。

    python tools/overnight_queue.py            # 前台（调试用）
    python tools/overnight_queue.py --dry-run  # 只打印将要执行的命令
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = r"E:\挑战赛"
PY = sys.executable
CACHE = os.path.join(ROOT, "cache")
FOLDS = os.path.join(ROOT, "04_新一轮", "00_规则", "patient_folds_5f.csv")
OUT = os.path.join(ROOT, "runs", "overnight")
LOG = os.path.join(OUT, "queue.log")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# 训练基础参数（与部署配方一致，除了显式说明的差异）
BASE = [
    "--arch", "convnext_tiny", "--img-size", "512", "--cache-size", "512",
    "--epochs", "12", "--batch-size", "16",
    "--lr-backbone", "1e-4", "--lr-head", "1e-3", "--weight-decay", "0.1",
    "--warmup-epochs", "2", "--label-smooth", "0.03", "--mixup", "0.2", "--cutmix", "0.2",
    "--drop-path", "0.15", "--ema", "0", "--workers", "0", "--amp", "bf16",
    "--patient-smooth", "0.3", "--head-type", "lse", "--lse-alpha", "0.25", "--lse-tau", "1.0",
    "--erase-prob", "0.25", "--rbc-prob", "0.25", "--rot-deg", "7", "--scale-min", "0.75",
]
KD = ["--distill-csv", os.path.join(ROOT, "reports", "teacher_soft_full_a.csv"),
      "--distill-weight", "0.5"]

ENV = dict(os.environ)
ENV["PYTHONPATH"] = os.path.join(ROOT, ".pylibs") + os.pathsep + os.path.join(ROOT, "src")
ENV["TORCH_HOME"] = os.path.join(ROOT, ".torch_home")
ENV["CXR_DATA"] = os.path.join(ROOT, "cxr_first")
ENV["PYTHONIOENCODING"] = "utf-8"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run(cmd: list[str], tag: str, dry: bool = False) -> bool:
    log(f"RUN {tag}: {' '.join(os.path.basename(c) if i == 0 else c for i, c in enumerate(cmd))[:160]}")
    if dry:
        return True
    t0 = time.time()
    try:
        with open(os.path.join(OUT, f"{tag}.log"), "a", encoding="utf-8", errors="replace") as fh:
            p = subprocess.run(cmd, cwd=ROOT, env=ENV, stdout=fh, stderr=subprocess.STDOUT)
        ok = p.returncode == 0
        log(f"{'OK ' if ok else 'FAIL'} {tag}  rc={p.returncode}  {time.time()-t0:.0f}s")
        return ok
    except Exception as exc:                                  # noqa: BLE001
        log(f"FAIL {tag}  {exc}")
        return False


def complete(run_dir: str) -> bool:
    """一次训练是否完整：history.json 满 12 轮且 last.pt 在。"""
    h = os.path.join(run_dir, "history.json")
    c = os.path.join(run_dir, "last.pt")
    if not (os.path.exists(h) and os.path.exists(c)):
        return False
    try:
        d = json.load(open(h, encoding="utf-8"))
        hist = d.get("history", d) if isinstance(d, dict) else d
        return isinstance(hist, list) and len(hist) >= 12
    except Exception:
        return False


def status(stage: str, payload: dict) -> None:
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "status.json")
    cur = {}
    if os.path.exists(path):
        try:
            cur = json.load(open(path, encoding="utf-8"))
        except Exception:
            cur = {}
    cur[stage] = payload
    cur["_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(cur, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-oof", action="store_true")
    ap.add_argument("--skip-repeat", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    log("=" * 70)
    log("夜间队列启动")
    log(f"折表 {FOLDS}  存在={os.path.exists(FOLDS)}")
    log(f"缓存 {CACHE}  存在={os.path.isdir(CACHE)}")
    if not os.path.exists(FOLDS):
        raise SystemExit("折表缺失，先跑 tools/make_patient_folds.py")
    if not os.path.isdir(CACHE):
        raise SystemExit("缓存缺失，先跑 src/build_cache.py --sizes 512")

    # ---------------- S0 指标工具自检 ----------------
    ok = run([PY, os.path.join(ROOT, "tools", "study_metric.py"),
              os.path.join(ROOT, "runs", "orig_valid.csv")], "S0_metric_smoke", a.dry_run)
    status("S0_metric", {"ok": ok})

    # ---------------- S1-5 五折 OOF ----------------
    if not a.skip_oof:
        for f in range(5):
            other = ",".join(str(i) for i in range(5) if i != f)
            run_dir = os.path.join(ROOT, "runs", f"oof_f{f}")
            ckpt = os.path.join(run_dir, "last.pt")
            tag = f"S1_oof_f{f}_train"
            if complete(run_dir):
                log(f"SKIP {tag}（已完成 12 轮）")
                status(tag, {"skipped": True})
            else:
                # no-KD（教师软标签在全 train 上训过，会泄漏外层留出患者）
                cmd = [PY, os.path.join(ROOT, "src", "train.py")] + BASE + [
                    "--seed", "42",
                    "--fold-csv", FOLDS, "--valid-fold", str(f), "--train-folds", other,
                    "--out-dir", run_dir, "--tag", f"oof_f{f}",
                ]
                ok = run(cmd, tag, a.dry_run)
                status(tag, {"ok": ok})
                if not ok:
                    log(f"折 {f} 训练失败，跳过该折的预测")
                    continue
            # 逐折预测
            pred_base = os.path.join(ROOT, "runs", f"oof_f{f}")
            if os.path.exists(pred_base + "_study.csv"):
                log(f"SKIP S2_oof_f{f}_predict（已存在）")
            else:
                run([PY, os.path.join(ROOT, "tools", "predict_fold.py"),
                     "--ckpt", ckpt, "--fold-csv", FOLDS, "--fold", str(f),
                     "--img-size", "544", "--cache-size", "512",
                     "--out-prefix", pred_base], f"S2_oof_f{f}_predict", a.dry_run)

    # ---------------- S7-8 受控重复 ----------------
    if not a.skip_repeat:
        for name in ("repA42", "repB42"):
            run_dir = os.path.join(ROOT, "runs", name)
            tag = f"S7_{name}_train"
            if complete(run_dir):
                log(f"SKIP {tag}（已完成 12 轮）")
                status(tag, {"skipped": True})
                continue
            cmd = [PY, os.path.join(ROOT, "src", "train.py")] + BASE + KD + [
                "--seed", "42", "--out-dir", run_dir, "--tag", name,
            ]
            ok = run(cmd, tag, a.dry_run)
            status(tag, {"ok": ok})

    log("队列结束（剩余汇总在 8:00 验收时执行）")


if __name__ == "__main__":
    main()
