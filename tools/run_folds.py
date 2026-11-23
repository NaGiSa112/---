"""通用折队列：跑 N 折训练 + 逐折预测，对照组是 runs/oof_f*。

    python tools/run_folds.py --tag blend48 --args "--label-blend-48 0.15"
    python tools/run_folds.py --tag asym --args "--extra-channel asym"

设计要点
--------
* 断点续跑：某折的 history.json 满 12 轮就跳过训练，逐折预测存在就跳过预测。
* 与 `runs/oof_f*` 可直接配对：overnight_queue 跑的五折是同折表、同 seed、no-KD、
  固定 last12，只要本脚本不额外加 KD/EMA，两边就是严格配对。
* 失败不重试、继续下一折，最后在日志里留痕。
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time

ROOT = r"E:\挑战赛"
PY = sys.executable
FOLDS = os.path.join(ROOT, "04_新一轮", "00_规则", "patient_folds_5f.csv")
ENV = dict(os.environ)
ENV["PYTHONPATH"] = os.path.join(ROOT, ".pylibs") + os.pathsep + os.path.join(ROOT, "src")
ENV["TORCH_HOME"] = os.path.join(ROOT, ".torch_home")
ENV["CXR_DATA"] = os.path.join(ROOT, "cxr_first")
ENV["PYTHONIOENCODING"] = "utf-8"

BASE = [
    "--arch", "convnext_tiny", "--img-size", "512", "--cache-size", "512",
    "--epochs", "12", "--batch-size", "16",
    "--lr-backbone", "1e-4", "--lr-head", "1e-3", "--weight-decay", "0.1",
    "--warmup-epochs", "2", "--label-smooth", "0.03", "--mixup", "0.2", "--cutmix", "0.2",
    "--drop-path", "0.15", "--ema", "0", "--workers", "0", "--amp", "bf16",
    "--patient-smooth", "0.3", "--head-type", "lse", "--lse-alpha", "0.25", "--lse-tau", "1.0",
    "--erase-prob", "0.25", "--rbc-prob", "0.25", "--rot-deg", "7", "--scale-min", "0.75",
]

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="运行前缀，产物在 runs/<tag>_f{fold}/")
    ap.add_argument("--args", default="", help="额外训练参数（引号内的整串，仅前台调试用）")
    ap.add_argument("--label-blend-48", type=float, default=0.0)
    ap.add_argument("--extra-channel", default="", choices=["", "asym"])
    ap.add_argument("--init-weights", default="")
    ap.add_argument("--patient-smooth-c1", type=float, default=-1.0)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    extra = shlex.split(a.args) if a.args else []
    if a.label_blend_48 > 0:
        extra += ["--label-blend-48", str(a.label_blend_48)]
    if a.extra_channel:
        extra += ["--extra-channel", a.extra_channel]
    if a.init_weights:
        extra += ["--init-weights", a.init_weights]
    if a.patient_smooth_c1 >= 0:
        extra += ["--patient-smooth-c1", str(a.patient_smooth_c1)]
    out = os.path.join(ROOT, "runs", a.tag)
    os.makedirs(out, exist_ok=True)
    log_path = os.path.join(out, "queue.log")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def complete(run_dir: str) -> bool:
        h = os.path.join(run_dir, "history.json")
        if not (os.path.exists(h) and os.path.exists(os.path.join(run_dir, "last.pt"))):
            return False
        try:
            d = json.load(open(h, encoding="utf-8"))
            hist = d.get("history", d) if isinstance(d, dict) else d
            return isinstance(hist, list) and len(hist) >= 12
        except Exception:
            return False

    def run(cmd, tag):
        log(f"RUN {tag}")
        if a.dry_run:
            return True
        t0 = time.time()
        with open(os.path.join(out, f"{tag}.log"), "a", encoding="utf-8", errors="replace") as fh:
            rc = subprocess.run(cmd, cwd=ROOT, env=ENV, stdout=fh,
                                stderr=subprocess.STDOUT).returncode
        log(f"{'OK ' if rc == 0 else 'FAIL'} {tag} rc={rc} {time.time()-t0:.0f}s")
        return rc == 0

    log("=" * 60)
    log(f"队列启动 tag={a.tag} extra={extra}")
    for f in [int(x) for x in a.folds.split(",") if x.strip()]:
        other = ",".join(str(i) for i in range(5) if i != f)
        run_dir = os.path.join(ROOT, "runs", f"{a.tag}_f{f}")
        tag = f"{a.tag}_f{f}_train"
        if complete(run_dir):
            log(f"SKIP {tag}（已完成 12 轮）")
        else:
            ok = run([PY, os.path.join(ROOT, "src", "train.py")] + BASE + extra + [
                "--seed", "42", "--fold-csv", FOLDS, "--valid-fold", str(f),
                "--train-folds", other, "--out-dir", run_dir, "--tag", f"{a.tag}_f{f}",
            ], tag)
            if not ok:
                log(f"折 {f} 训练失败，跳过预测")
                continue
        pred_base = os.path.join(ROOT, "runs", f"{a.tag}_f{f}")
        if os.path.exists(pred_base + "_study.csv"):
            log(f"SKIP {a.tag}_f{f}_predict")
        else:
            run([PY, os.path.join(ROOT, "tools", "predict_fold.py"),
                 "--ckpt", os.path.join(run_dir, "last.pt"), "--fold-csv", FOLDS,
                 "--fold", str(f), "--img-size", "544", "--cache-size", "512",
                 "--out-prefix", pred_base], f"{a.tag}_f{f}_predict")
    log("队列结束")


if __name__ == "__main__":
    main()
