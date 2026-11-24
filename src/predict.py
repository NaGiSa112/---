"""Submission generation.

    python predict.py --split test --out submission.csv

Loads the ensemble listed in model/deploy.json (see cxr/deploy.py), runs the
deterministic inference path with flip TTA, and writes the submission csv.
`cost.py` imports the same EnsemblePredictor, so both produce identical
predictions (competition rule 7).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cxr.cache import label_matrix, read_split  # noqa: E402
from cxr.config import LABELS, NUM_CLASSES, REPORTS  # noqa: E402
from cxr.deploy import DEFAULT_DEPLOY, EnsemblePredictor, thresholds_for  # noqa: E402
from cxr.infer import write_submission  # noqa: E402
from cxr.metrics import mean_auc, patient_smooth  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEPLOY = os.path.join(ROOT, DEFAULT_DEPLOY)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy", default=DEPLOY)
    ap.add_argument("--split", default="test", choices=["test", "valid", "train"])
    ap.add_argument("--out", default=os.path.join(ROOT, "submissions", "submission.csv"))
    ap.add_argument("--probs-out", default="")
    ap.add_argument("--patient-smooth", action="store_true",
                    help="average a patient's images (rules supplement section 7); "
                         "disabled by default because it loses ~0.02 AUC on valid")
    ap.add_argument("--no-pred-label", action="store_true")
    ap.add_argument("--format", default="official", choices=["official", "main", "alt"],
                    help="official = 每行一个 (Subject_id, Study_id, Predict_class, Probability)")
    ap.add_argument("--wide-out", default="",
                    help="可选：另存一份逐图宽表（供 cost.py 逐样本核对）")
    ap.add_argument("--lat-weight", type=float, default=0.6)
    ap.add_argument("--shrink", type=float, default=0.85)
    ap.add_argument("--shrink-by-class", default="",
                    help='逐类收缩，形如 "2:0.45" 或 "2:0.45,5:0.9"；留空=全部用 --shrink')
    ap.add_argument("--batch-size", type=int, default=1,
                    help="1 keeps the numerics bit-identical to cost.py (rule 7 consistency)")
    a = ap.parse_args()

    pred = EnsemblePredictor(a.deploy)
    print(f"[predict] ensemble: {[m['name'] for m in pred.models]} amp={pred.amp}")
    df = read_split(a.split)
    probs = pred.predict_frame(df, a.split, batch_size=a.batch_size)
    if a.patient_smooth:
        probs = patient_smooth(df, probs)

    if "Predict_class" in df.columns:
        y = label_matrix(df)
        auc, aucs = mean_auc(y, probs)
        auc_s, _ = mean_auc(y, patient_smooth(df, probs))
        print(f"[{a.split}] mean AUC = {auc:.4f}  (patient smoothed {auc_s:.4f})")
        print("  per class:", {LABELS[i]: round(float(aucs[i]), 4) for i in range(NUM_CLASSES)})

    if a.probs_out:
        out = df[["Dicom_id", "Subject_id", "Study_id"]].copy()
        for k in range(NUM_CLASSES):
            out[f"p{k+1}"] = probs[:, k]
        os.makedirs(os.path.dirname(os.path.abspath(a.probs_out)), exist_ok=True)
        out.to_csv(a.probs_out, index=False)
        print("wrote", a.probs_out)

    thr = thresholds_for(a.deploy)
    if not thr:
        print("[warn] no thresholds.json found, falling back to 0.5")
    sbc = {}
    if a.shrink_by_class:
        for part in a.shrink_by_class.split(","):
            part = part.strip()
            if not part:
                continue
            k, v = part.split(":")
            sbc[str(int(k))] = float(v)
    sub = write_submission(df, probs, a.out, thr, pred_label=not a.no_pred_label,
                           fmt=("official" if a.format == "official" else a.format),
                           lat_weight=a.lat_weight, shrink=a.shrink,
                           shrink_by_class=sbc)
    print(f"wrote {a.out} rows={len(sub)} cols={list(sub.columns)}")
    if a.wide_out:
        wide = write_submission(df, probs, a.wide_out, thr,
                                pred_label=not a.no_pred_label, fmt="main")
        print(f"wrote {a.wide_out} rows={len(wide)} cols={list(wide.columns)}")
    return sub


if __name__ == "__main__":
    main()
