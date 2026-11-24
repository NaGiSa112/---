"""从训练得到的 checkpoint 生成 valid 逐图概率（云端与本地通用）。

train.py 只存 best.pt/last.pt/history.json，不导出预测；这个脚本补上那一步，
产出的 CSV 可直接喂给 `tools/e1_score.py` 做 E1/E2 判读。

    python tools/make_probs_from_ckpt.py \
        --ckpt /root/autodl-tmp/cxr/runs/caformer_f2/last.pt \
        --arch caformer_s18.sail_in22k_ft_in1k --img 544 \
        --out  /root/autodl-tmp/cxr/runs/caformer_f2/valid_probs.csv

也可以一次处理多个 run：
    python tools/make_probs_from_ckpt.py --runs-dir /root/autodl-tmp/cxr/runs
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

ROOT = os.environ.get("CXR_ROOT_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# 已知架构的输入尺寸（与训练时一致）
DEFAULT_IMG = {
    "caformer_s18.sail_in22k_ft_in1k": 544,
    "swinv2_tiny_window16_256.ms_in1k": 512,
    "convnext_tiny": 544,
}


def run_one(ckpt: str, arch: str, img: int, out_csv: str) -> int:
    spec = {
        "amp": "bf16", "fusion": "arithmetic", "cpu_threads": 4, "graph_tiny": True,
        "members": [{
            "name": os.path.basename(os.path.dirname(os.path.abspath(ckpt))),
            "ckpt": os.path.abspath(ckpt),
            "img_size": int(img), "cache_size": 512, "norm": "per_image",
            "tta": True, "weight": 1.0, "lse_alpha": 0.25,
        }],
    }
    spec_path = out_csv.replace(".csv", "_deploy.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    json.dump(spec, open(spec_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    cmd = [sys.executable, os.path.join(ROOT, "src", "predict.py"),
           "--split", "valid", "--batch-size", "1", "--deploy", spec_path,
           "--lat-weight", "0.6", "--shrink", "0.85",
           "--out", out_csv.replace("probs", "sub"), "--probs-out", out_csv]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(ROOT, "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    print("  $", " ".join(cmd[:6]), "...")
    r = subprocess.run(cmd, cwd=ROOT, env=env)
    print(f"  rc={r.returncode}  →  {out_csv}")
    return r.returncode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--arch", default="")
    ap.add_argument("--img", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--runs-dir", default="")
    a = ap.parse_args()

    if a.runs_dir:
        runs = a.runs_dir
        todo = []
        for name in sorted(os.listdir(runs)):
            d = os.path.join(runs, name)
            if not os.path.isdir(d):
                continue
            ck = os.path.join(d, "last.pt")
            cfg = os.path.join(d, "config.json")
            if not (os.path.exists(ck) and os.path.exists(cfg)):
                continue
            c = json.load(open(cfg, encoding="utf-8"))
            arch = c.get("arch", "")
            img = int(c.get("img_size", DEFAULT_IMG.get(arch, 544)))
            out = os.path.join(d, "valid_probs.csv")
            if os.path.exists(out):
                print(f"  {name}: 已有 valid_probs.csv，跳过")
                continue
            todo.append((ck, arch, img, out))
        for ck, arch, img, out in todo:
            print(f"\n=== {os.path.basename(os.path.dirname(ck))}  arch={arch} img={img}")
            run_one(ck, arch, img, out)
        return

    if not (a.ckpt and a.out):
        ap.error("需要 --ckpt 与 --out，或使用 --runs-dir")
    arch = a.arch
    img = a.img or DEFAULT_IMG.get(arch, 544)
    if not arch:
        cfg = os.path.join(os.path.dirname(os.path.abspath(a.ckpt)), "config.json")
        if os.path.exists(cfg):
            arch = json.load(open(cfg, encoding="utf-8")).get("arch", "")
            img = a.img or int(json.load(open(cfg, encoding="utf-8")).get("img_size", img))
    print(f"  arch={arch}  img={img}")
    sys.exit(run_one(a.ckpt, arch, img, a.out))


if __name__ == "__main__":
    main()
