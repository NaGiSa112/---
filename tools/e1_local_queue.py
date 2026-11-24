"""E1 本地顺序队列：CAFormer-S18 fold2 → SwinV2-T fold2，各自跑完自动算指标。

用法
----
    python tools/e1_local_queue.py            # 跑两条线（约 3.1 小时）
    python tools/e1_local_queue.py --only caformer
    python tools/e1_local_queue.py --only swinv2

每个训练用子进程 + 日志文件，跑完立刻用 valid 逐图概率算检查级宏 AUC，
写入 reports/E1_本地结果.json，方便第二天直接看结论。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = r"E:\挑战赛"
# 本地也走本地权重（避免 timm 去连 huggingface）
os.environ.setdefault("CXR_WEIGHTS_DIR", os.path.join(ROOT, ".weights"))
os.environ.setdefault("PYTHONUNBUFFERED", "1")
LOGDIR = os.path.join(ROOT, "runs", "e1_local")
SUMMARY = os.path.join(ROOT, "reports", "E1_本地结果.json")
FOLD = os.path.join(ROOT, "04_新一轮", "00_规则", "patient_folds_5f.csv")
PCOLS = list(range(1, 11))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

COMMON = [
    "--epochs", "12", "--batch-size", "16",
    "--lr-backbone", "1e-4", "--lr-head", "1e-3", "--weight-decay", "0.1",
    "--warmup-epochs", "2", "--label-smooth", "0.03", "--mixup", "0.2", "--cutmix", "0.2",
    "--drop-path", "0.15", "--ema", "0", "--workers", "0", "--amp", "bf16",
    "--patient-smooth", "0.3", "--head-type", "lse", "--lse-alpha", "0.25", "--lse-tau", "1.0",
    "--erase-prob", "0.25", "--rbc-prob", "0.25", "--rot-deg", "7", "--scale-min", "0.75",
    "--seed", "42", "--fold-csv", FOLD, "--valid-fold", "2", "--train-folds", "0,1,3,4",
]

JOBS = {
    "caformer": {
        "arch": "caformer_s18.sail_in22k_ft_in1k",
        "img": "544",
        "out": os.path.join(ROOT, "runs", "caformer_f2"),
    },
    "swinv2": {
        "arch": "swinv2_tiny_window16_256.ms_in1k",
        "img": "512",
        "out": os.path.join(ROOT, "runs", "swinv2_f2"),
    },
}


def parse_label(s) -> np.ndarray:
    v = np.zeros(10, np.float32)
    s = str(s).strip()
    if s == "" or s.lower() == "nan":
        return v
    cs = [c.strip() for c in s.split(";") if c.strip()]
    if cs == ["10"]:
        v[9] = 1.0
        return v
    for c in cs:
        i = int(c) - 1
        if i < 9:
            v[i] = 1.0
    return v


def auc(y, p) -> float:
    o = np.argsort(p)
    r = np.empty(len(p), float)
    r[o] = np.arange(1, len(p) + 1)
    pos = y > 0
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def score(probs_csv: str) -> dict:
    va = pd.read_csv(os.path.join(ROOT, "cxr_first", "valid.csv"))
    Y = np.stack([parse_label(s) for s in va.Predict_class])
    lab = pd.DataFrame(Y, columns=PCOLS)
    lab["Subject_id"] = va.Subject_id.values
    lab["Study_id"] = va.Study_id.values
    lab = lab.groupby(["Subject_id", "Study_id"])[PCOLS].max()

    d = pd.read_csv(probs_csv).merge(va[["Dicom_id", "ViewPosition"]], on="Dicom_id", how="left")
    w = np.where(np.isin(d["ViewPosition"].fillna("NA").values, ["LATERAL", "LL"]), 0.6, 1.0)
    for c in [f"p{k}" for k in range(1, 11)]:
        d[c] = d[c] * w
    g = d.groupby(["Subject_id", "Study_id"], as_index=False)[[f"p{k}" for k in range(1, 11)]].sum()
    ws = pd.Series(w).groupby([d["Subject_id"].values, d["Study_id"].values]).sum()
    cols = [f"p{k}" for k in range(1, 11)]
    g[cols] = g[cols].div(ws.values[:, None])
    pm = g.groupby("Subject_id")[cols].transform("mean").values
    g[cols] = 0.85 * g[cols].values + 0.15 * pm
    g.columns = ["Subject_id", "Study_id"] + PCOLS
    g = g.set_index(["Subject_id", "Study_id"]).loc[lab.index]
    per = [auc(lab[c].to_numpy(), g[c].to_numpy()) for c in PCOLS]
    return {"macro": float(np.nanmean(per)), "per_class": per, "n_study": len(lab)}


def predict_valid(ckpt: str, out_csv: str, arch: str, img: str) -> int:
    """用训练得到的 checkpoint 跑一次 valid 单成员推理，落盘逐图概率。

    train.py 本身不导出预测，所以这里复用 predict.py（与交付同一套推理代码）。
    """
    spec = {
        "amp": "bf16", "fusion": "arithmetic", "cpu_threads": 4, "graph_tiny": True,
        "members": [{
            "name": os.path.basename(os.path.dirname(ckpt)),
            "ckpt": os.path.abspath(ckpt),
            "img_size": int(img), "cache_size": 512, "norm": "per_image",
            "tta": True, "weight": 1.0, "lse_alpha": 0.25,
        }],
    }
    spec_path = os.path.join(LOGDIR, f"_spec_{os.path.basename(os.path.dirname(ckpt))}.json")
    json.dump(spec, open(spec_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    cmd = [sys.executable, os.path.join(ROOT, "src", "predict.py"),
           "--split", "valid", "--batch-size", "1", "--deploy", spec_path,
           "--lat-weight", "0.6", "--shrink", "0.85",
           "--out", out_csv.replace("probs", "sub"), "--probs-out", out_csv]
    log = os.path.join(LOGDIR, "predict.log")
    with open(log, "a", encoding="utf-8") as f:
        f.write("\n$ " + " ".join(cmd) + "\n")
        f.flush()
        rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                            env=dict(os.environ,
                                     PYTHONPATH=os.path.join(ROOT, ".pylibs") + os.pathsep
                                     + os.path.join(ROOT, "src"),
                                     PYTHONIOENCODING="utf-8",
                                     CXR_DATA=os.path.join(ROOT, "cxr_first"),
                                     CXR_CACHE=os.path.join(ROOT, "cache"))).returncode
    return rc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", choices=["", "caformer", "swinv2"])
    a = ap.parse_args()
    os.makedirs(LOGDIR, exist_ok=True)

    order = ["caformer", "swinv2"] if not a.only else [a.only]
    results = {}
    if os.path.exists(SUMMARY):
        results = json.load(open(SUMMARY, encoding="utf-8"))

    for name in order:
        job = JOBS[name]
        outdir = job["out"]
        hist = os.path.join(outdir, "history.json")
        if os.path.exists(hist):
            try:
                h = json.load(open(hist, encoding="utf-8"))
                hl = h.get("history", h)
                if len(hl) >= 12:
                    print(f"  {name}: 已完成 12 轮，跳过训练")
                    probs = os.path.join(outdir, "valid_probs.csv")
                    if os.path.exists(probs):
                        results[name] = score(probs)
                    continue
            except Exception:
                pass

        cmd = [sys.executable, os.path.join(ROOT, "src", "train.py"),
               "--arch", job["arch"], "--img-size", job["img"], "--cache-size", "512",
               "--out-dir", outdir, "--tag", f"{name}_f2"] + COMMON
        log = os.path.join(LOGDIR, f"{name}.log")
        print(f"\n=== [{time.strftime('%H:%M:%S')}] 开始训练 {name} → 日志 {log}")
        t0 = time.time()
        with open(log, "w", encoding="utf-8") as f:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                                env=dict(os.environ,
                                         PYTHONPATH=os.path.join(ROOT, ".pylibs") + os.pathsep
                                         + os.path.join(ROOT, "src"),
                                         PYTHONIOENCODING="utf-8",
                                         CXR_DATA=os.path.join(ROOT, "cxr_first"),
                                         CXR_CACHE=os.path.join(ROOT, "cache"))).returncode
        dt = (time.time() - t0) / 60
        print(f"    {name}: rc={rc}，用时 {dt:.1f} 分钟")
        probs = os.path.join(outdir, "valid_probs.csv")
        if rc == 0 and not os.path.exists(probs):
            ck = os.path.join(outdir, "last.pt")
            if os.path.exists(ck):
                print("    训练完成，导出 valid 逐图概率…")
                prc = predict_valid(ck, probs, job["arch"], job["img"])
                print(f"    predict rc={prc}")
        if rc == 0 and os.path.exists(probs):
            s = score(probs)
            s["train_minutes"] = round(dt, 1)
            results[name] = s
            print(f"    宏 AUC = {s['macro']:.6f}（门槛 0.8200）"
                  f" → {'✅ 过' if s['macro'] >= 0.82 else '❌ 未过'}")
        json.dump(results, open(SUMMARY, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print("\n=== 汇总")
    for k, v in results.items():
        print(f"  {k:10s} 宏 {v['macro']:.6f}  " +
              " ".join(f"{x:.3f}" for x in v["per_class"]))
    print(f"  [wrote] {os.path.relpath(SUMMARY, ROOT)}")


if __name__ == "__main__":
    main()
