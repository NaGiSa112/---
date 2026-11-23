"""接力器：等某个队列跑完后，自动启动下一个实验。

为什么需要它
------------
GPU 只有一个，训练必须串行。人在 22K 跑完的那一刻未必在线，所以用这个脚本轮询：
当被监视的 tag 不再有进程、且它的目标折都已完成（或已明确失败退出）时，
自动启动下一个队列。

安全边界
--------
* 只做"启动下一个"，不改动正在运行的源码或产物；
* 被监视 tag 若长时间无进展（默认 90 分钟）则放弃并记录，不无限等待；
* 启动前调用 tools/verify_predictions.py 检查已完成的预测（仅记录，不阻塞）；
* 全程写 runs/<tag>/chain.log。

    python tools/chain_run.py --watch init22k --tag pat_c1 --folds 0,1 \
        --patient-smooth-c1 0.10
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

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def log(path: str, msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def queue_state(tag: str) -> tuple[bool, float]:
    """返回 (是否已宣布结束, 静默秒数)。

    静默时间取「queue.log 与所有 <tag>_f*_train.log / *predict.log 的最新修改时间」——
    queue.log 只在阶段边界写一行，训练过程中它是静的，但训练日志一直在滚。
    只看 queue.log 会把"正在训练"误判成"已挂掉"（2026-11-20 已经踩过一次，导致两个训练并发）。
    """
    q = os.path.join(ROOT, "runs", tag, "queue.log")
    base = os.path.join(ROOT, "runs", tag)
    mtimes = []
    if os.path.exists(q):
        mtimes.append(os.path.getmtime(q))
    if os.path.isdir(base):
        for f in os.listdir(base):
            if f.endswith(".log") and ("_train.log" in f or "_predict.log" in f):
                mtimes.append(os.path.getmtime(os.path.join(base, f)))
    # 该 tag 自己的 run_dir 里的 history.json 也在训练中更新
    for f in range(5):
        h = os.path.join(ROOT, "runs", f"{tag}_f{f}", "history.json")
        if os.path.exists(h):
            mtimes.append(os.path.getmtime(h))
    idle = time.time() - max(mtimes) if mtimes else float("inf")
    finished = False
    if os.path.exists(q):
        txt = open(q, encoding="utf-8", errors="replace").read()
        finished = ("队列结束" in txt)
    return finished, idle


def fold_done(tag: str, f: int) -> bool:
    h = os.path.join(ROOT, "runs", f"{tag}_f{f}", "history.json")
    if not os.path.exists(h):
        return False
    try:
        d = json.load(open(h, encoding="utf-8"))
        hist = d.get("history", d) if isinstance(d, dict) else d
        return isinstance(hist, list) and len(hist) >= 12
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", required=True, help="被监视的 tag")
    ap.add_argument("--watch-folds", default="0,1")
    ap.add_argument("--tag", required=True, help="要启动的新 tag")
    ap.add_argument("--folds", default="0,1")
    ap.add_argument("--patient-smooth-c1", type=float, default=-1.0)
    ap.add_argument("--init-weights", default="")
    ap.add_argument("--label-blend-48", type=float, default=0.0)
    ap.add_argument("--poll", type=int, default=120, help="轮询间隔秒")
    ap.add_argument("--max-wait", type=int, default=5400, help="最长等待秒（默认 90 分钟）")
    a = ap.parse_args()
    logp = os.path.join(ROOT, "runs", a.tag, "chain.log")
    os.makedirs(os.path.join(ROOT, "runs", a.tag), exist_ok=True)
    log(logp, f"接力器启动：等待 {a.watch} 折{a.watch_folds} 结束 → 启动 {a.tag} 折{a.folds}")

    t0 = time.time()
    wf = [int(x) for x in a.watch_folds.split(",") if x.strip()]
    while True:
        done = all(fold_done(a.watch, f) for f in wf)
        finished, idle = queue_state(a.watch)
        if done:
            log(logp, f"{a.watch} 折{wf} 全部完成 12 轮 → 开始接力")
            break
        if finished:
            log(logp, f"{a.watch} 队列已宣布结束但折{wf}未全部满 12 轮（可能失败）"
                      f" → 不启动 {a.tag}，请人工确认")
            return
        if idle > 3600:
            log(logp, f"{a.watch} 已 {idle/60:.0f} 分钟无任何日志更新，疑似挂掉"
                      f" → **不启动** {a.tag}（避免与可能还活着的训练并发），请人工确认")
            return
        if time.time() - t0 > a.max_wait:
            log(logp, f"等待超过 {a.max_wait//60} 分钟 → 放弃接力")
            return
        time.sleep(a.poll)

    cmd = [PY, os.path.join(ROOT, "tools", "run_folds.py"), "--tag", a.tag, "--folds", a.folds]
    if a.patient_smooth_c1 >= 0:
        cmd += ["--patient-smooth-c1", str(a.patient_smooth_c1)]
    if a.init_weights:
        cmd += ["--init-weights", a.init_weights]
    if a.label_blend_48 > 0:
        cmd += ["--label-blend-48", str(a.label_blend_48)]
    log(logp, "启动: " + " ".join(os.path.basename(c) if i == 0 else c
                                  for i, c in enumerate(cmd))[:200])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(ROOT, ".pylibs") + os.pathsep + os.path.join(ROOT, "src")
    env["TORCH_HOME"] = os.path.join(ROOT, ".torch_home")
    env["CXR_DATA"] = os.path.join(ROOT, "cxr_first")
    env["PYTHONIOENCODING"] = "utf-8"
    with open(os.path.join(ROOT, "runs", a.tag, "runner.log"), "a",
              encoding="utf-8", errors="replace") as fh:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    log(logp, f"{a.tag} 结束 rc={rc}")
    if rc == 0:
        chk_log = os.path.join(ROOT, "runs", a.tag, "verify.log")
        with open(chk_log, "a", encoding="utf-8", errors="replace") as fh:
            rc2 = subprocess.run([PY, os.path.join(ROOT, "tools", "verify_predictions.py"),
                                  "--tag", a.tag, "--folds", a.folds], cwd=ROOT,
                                 env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
        log(logp, "完整性检查: " + ("通过" if rc2 == 0 else "未通过") + f"（见 {os.path.relpath(chk_log, ROOT)}）")


if __name__ == "__main__":
    main()
