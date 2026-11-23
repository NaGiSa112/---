"""Evaluate checkpoints on the official validation split and fit decision thresholds."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cxr.cache import label_matrix, read_split  # noqa: E402
from cxr.config import LABELS, LABEL_NAMES, NUM_CLASSES, REPORTS  # noqa: E402
from cxr.infer import (PathDataset, load_checkpoint, predict_probs,  # noqa: E402
                       thresholds_from_valid)
from cxr.metrics import best_f1_threshold, f1_at, mean_auc, patient_smooth  # noqa: E402


def evaluate_ckpt(ckpt_paths, split="valid", tta=True, batch_size=16, amp="bf16",
                  img_size=384, cache_size=448, cv_fold=None, cv_folds=0, seed=42):
    if cv_folds and cv_folds > 1:
        # patient-grouped CV: the holdout is a fold of train.csv, so read the train cache
        from cxr.train import patient_fold_split
        _, df = patient_fold_split(read_split("train"), cv_fold, cv_folds, seed)
        split = "train"
        print(f"[evaluate] CV mode: fold {cv_fold}/{cv_folds} holdout, "
              f"{len(df)} imgs / {df.Subject_id.nunique()} patients")
    else:
        df = read_split(split)
    y = label_matrix(df)
    probs = []
    for c in ckpt_paths:
        model, _ = load_checkpoint(c, "cuda" if torch.cuda.is_available() else "cpu")
        ds = PathDataset(df, split, img_size=img_size, cache_size=cache_size)
        probs.append(predict_probs(model, ds, tta=tta, batch_size=batch_size, amp=amp))
        del model
        torch.cuda.empty_cache()
    p = np.mean(probs, axis=0)
    return df, y, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--out-prefix", default=os.path.join(REPORTS, "valid"))
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--cache-size", type=int, default=448)
    ap.add_argument("--cv-fold", type=int, default=None)
    ap.add_argument("--cv-folds", type=int, default=0)
    a = ap.parse_args()

    df, y, p = evaluate_ckpt(a.ckpt, a.split, tta=not a.no_tta,
                             img_size=a.img_size, cache_size=a.cache_size,
                             cv_fold=a.cv_fold, cv_folds=a.cv_folds)
    auc, aucs = mean_auc(y, p)
    ps = patient_smooth(df, p)
    auc_s, aucs_s = mean_auc(y, ps)
    print(f"mean AUC (image level)   = {auc:.4f}")
    print(f"mean AUC (patient smooth)= {auc_s:.4f}")

    rows = []
    for k in range(NUM_CLASSES):
        thr, f1 = best_f1_threshold(y[:, k], ps[:, k])
        rows.append({"label": LABELS[k], "name": LABEL_NAMES[LABELS[k]],
                     "n_pos": int(y[:, k].sum()),
                     "auc": round(float(aucs[k]), 4),
                     "auc_patient": round(float(aucs_s[k]), 4),
                     "f1": round(f1, 4), "threshold": round(thr, 6)})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    os.makedirs(REPORTS, exist_ok=True)
    out = df[["Dicom_id", "Subject_id", "Study_id"]].copy()
    for k in range(NUM_CLASSES):
        out[f"p{k+1}"] = p[:, k]
    out.to_csv(a.out_prefix + "_probs.csv", index=False)
    table.to_csv(a.out_prefix + "_per_class.csv", index=False)
    thr = thresholds_from_valid(a.out_prefix + "_probs.csv", a.out_prefix + "_thresholds.json")
    macro_f1 = float(np.mean([
        f1_at(y[:, k], ps[:, k], thr[f"p{k+1}"]) for k in range(NUM_CLASSES)]))
    print(f"macro F1 at F1-optimal thresholds (patient smoothed) = {macro_f1:.4f}")
    with open(a.out_prefix + "_summary.json", "w", encoding="utf-8") as f:
        json.dump({"ckpts": a.ckpt, "split": a.split, "mean_auc": auc,
                   "mean_auc_patient": auc_s, "macro_f1": macro_f1,
                   "per_class": rows, "thresholds": thr}, f, indent=2)


if __name__ == "__main__":
    main()
