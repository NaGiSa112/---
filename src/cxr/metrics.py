"""Evaluation utilities: AUC, thresholds, patient-level smoothing."""
from __future__ import annotations

import numpy as np


def roc_auc_single(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Rank-based ROC-AUC that handles ties (Mann-Whitney U)."""
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos = y_true == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)
    # average ranks over ties
    sorted_scores = y_score[order]
    i = 0
    n = len(sorted_scores)
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j - i > 1:
            avg = (i + 1 + j) / 2.0
            ranks[order[i:j]] = avg
        i = j
    sum_pos = ranks[pos].sum()
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def per_class_auc(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    return np.array([roc_auc_single(y_true[:, k], y_score[:, k]) for k in range(y_true.shape[1])])


def mean_auc(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, np.ndarray]:
    aucs = per_class_auc(y_true, y_score)
    return float(np.nanmean(aucs)), aucs


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Return (threshold, f1) maximising F1 for one binary column."""
    y_true = np.asarray(y_true).astype(np.int64)
    if y_true.sum() == 0:
        return 0.5, 0.0
    order = np.argsort(-y_score)
    s = y_score[order]
    t = y_true[order]
    tp = np.cumsum(t)
    fp = np.cumsum(1 - t)
    fn = t.sum() - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)
    i = int(np.argmax(f1))
    thr = float(np.nextafter(s[i], -np.inf))
    return thr, float(f1[i])


def f1_at(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> float:
    pred = (y_score >= thr).astype(np.int64)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1)


def patient_smooth(df, probs: np.ndarray, key: str = "Subject_id") -> np.ndarray:
    """Deterministic aggregation of a patient's predictions (rules section 7)."""
    out = probs.copy()
    idx = {}
    for i, s in enumerate(df[key].values):
        idx.setdefault(s, []).append(i)
    for s, rows in idx.items():
        if len(rows) > 1:
            out[rows] = probs[rows].mean(axis=0)
    return out


def stack_tta_logits(logits_list) -> np.ndarray:
    """Average sigmoid probabilities over TTA views."""
    probs = [1.0 / (1.0 + np.exp(-l)) for l in logits_list]
    return np.mean(probs, axis=0)
