"""
Evaluation metrics.

Reproduces the six headline metrics reported in Chen et al. (npj Health
Systems 2025) Figs 1 and reported in Methods § Experimental settings:

    AUROC, AUPRC, F1, precision, recall, specificity.

Plus the inferential machinery the paper applies on top:

    * DeLong test for AUROC comparison between two models on the same
      held-out set (Methods: "DeLong's test was applied").
    * 95% confidence intervals on AUROC and AUPRC by bootstrap
      (Methods: "95% CI for AUROC and AUPRC were estimated using
      bootstrapping").

Operating-point metrics (F1, precision, recall, specificity) use the
threshold that maximises F1 on the validation set, picked once and
held fixed when scoring the test set.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


# =====================================================================
@dataclass
class MetricResult:
    auroc:       float
    auprc:       float
    f1:          float
    precision:   float
    recall:      float
    specificity: float
    threshold:   float
    auroc_ci:    Tuple[float, float]
    auprc_ci:    Tuple[float, float]

    def to_dict(self) -> dict:
        return asdict(self)


# =====================================================================
def pick_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Operating point that maximises F1 on the supplied set (typically
    val). Used at test time to compute the operating-point metrics."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    f1s = (2 * precisions * recalls) / np.maximum(precisions + recalls, 1e-9)
    if len(thresholds) == 0:
        return 0.5
    best = int(np.nanargmax(f1s[:-1])) if len(f1s) > 1 else 0
    return float(thresholds[min(best, len(thresholds) - 1)])


# =====================================================================
def _specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / max(tn + fp, 1)


# =====================================================================
def _bootstrap_ci(
    y_true: np.ndarray, y_score: np.ndarray,
    metric_fn, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0,
) -> Tuple[float, float]:
    """Percentile-bootstrap (1-alpha) confidence interval for a metric
    that takes (y_true, y_score)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats_ = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        stats_.append(metric_fn(y_true[idx], y_score[idx]))
    stats_ = np.asarray(stats_)
    lo, hi = np.percentile(stats_, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# =====================================================================
def evaluate(
    y_true: np.ndarray, y_score: np.ndarray,
    threshold: Optional[float] = None,
    n_boot: int = 1000, seed: int = 0,
) -> MetricResult:
    """Full metric panel. If `threshold` is None, pick it on the input
    set (use this only for validation results)."""
    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)

    if threshold is None:
        threshold = pick_threshold(y_true, y_score)
    y_pred = (y_score >= threshold).astype(int)

    f1  = f1_score(y_true, y_pred, zero_division=0)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    spe = _specificity(y_true, y_pred)

    auroc_ci = _bootstrap_ci(y_true, y_score, roc_auc_score, n_boot, 0.05, seed)
    auprc_ci = _bootstrap_ci(y_true, y_score, average_precision_score,
                             n_boot, 0.05, seed + 1)

    return MetricResult(
        auroc=auroc, auprc=auprc,
        f1=f1, precision=pre, recall=rec, specificity=spe,
        threshold=float(threshold),
        auroc_ci=auroc_ci, auprc_ci=auprc_ci,
    )


# =====================================================================
# Decision curve analysis (Vickers & Elkin 2006). Net benefit quantifies
# the clinical value of acting on a model's predictions at a given
# threshold probability p_t, on the same scale as treat-all / treat-none
# defaults, so a curve that sits above both over the plausible threshold
# range demonstrates utility that AUROC/AUPRC cannot show.
#
#   NB(p_t) = TP/N - (FP/N) * (p_t / (1 - p_t))
#
# A patient is flagged when predicted probability >= p_t. The odds factor
# p_t/(1-p_t) is the exchange rate between false positives and true
# positives implied by choosing p_t as the action threshold.
# =====================================================================
@dataclass
class DecisionCurve:
    thresholds:    np.ndarray   # p_t grid
    net_benefit:   np.ndarray   # model NB at each p_t
    treat_all:     np.ndarray   # NB of treating everyone
    treat_none:    np.ndarray   # NB of treating no one (== 0)
    prevalence:    float

    def to_dict(self) -> dict:
        return {
            "thresholds":  self.thresholds.tolist(),
            "net_benefit": self.net_benefit.tolist(),
            "treat_all":   self.treat_all.tolist(),
            "treat_none":  self.treat_none.tolist(),
            "prevalence":  self.prevalence,
        }


def net_benefit(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> float:
    """Net benefit of acting on `y_score >= threshold` at the action
    threshold probability `threshold` (Vickers & Elkin 2006)."""
    y_true = np.asarray(y_true)
    n = len(y_true)
    if n == 0 or threshold >= 1.0:
        return 0.0
    flag = y_score >= threshold
    tp = int(np.sum(flag & (y_true == 1)))
    fp = int(np.sum(flag & (y_true == 0)))
    w = threshold / (1.0 - threshold)
    return tp / n - (fp / n) * w


def decision_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: Optional[Sequence[float]] = None,
) -> DecisionCurve:
    """Net-benefit curve for one model across a threshold-probability grid,
    with the treat-all and treat-none reference strategies."""
    y_true = np.asarray(y_true)
    if thresholds is None:
        thresholds = np.arange(0.01, 0.51, 0.01)
    thr = np.asarray(thresholds, dtype=float)
    prev = float(np.mean(y_true == 1))

    nb = np.array([net_benefit(y_true, y_score, float(t)) for t in thr])
    # treat-all: flag everyone -> TP/N = prevalence, FP/N = 1 - prevalence
    w = thr / (1.0 - thr)
    treat_all = prev - (1.0 - prev) * w
    treat_none = np.zeros_like(thr)
    return DecisionCurve(thr, nb, treat_all, treat_none, prev)


# =====================================================================
# DeLong test for two correlated AUROCs (paper uses this to compare
# MedHG-PS with each ML baseline on the same held-out set).
#
# Implementation: Sun & Xu (2014) fast algorithm.
# =====================================================================
def _compute_midrank(x: np.ndarray) -> np.ndarray:
    n = len(x)
    j = np.argsort(x, kind="mergesort")
    s = x[j]
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        k = i
        while k < n and s[k] == s[i]:
            k += 1
        avg = 0.5 * (i + k - 1) + 1.0
        t[j[i:k]] = avg
        i = k
    return t


def delong_roc_variance(
    y_true: np.ndarray, scores: Dict[str, np.ndarray]
) -> Tuple[Dict[str, float], np.ndarray]:
    """Compute AUROC and the covariance matrix between any number of
    models' AUROC estimates on the same labels using Sun & Xu's fast
    DeLong algorithm.

    Returns (auroc_per_model, covariance_matrix [n_models x n_models]).
    """
    pos = y_true == 1
    m, n = int(pos.sum()), int((~pos).sum())
    if m == 0 or n == 0:
        raise ValueError("DeLong requires both classes present.")
    names = list(scores.keys())
    K = len(names)

    tx = np.zeros((K, m))
    ty = np.zeros((K, n))
    tz = np.zeros((K, m + n))

    aucs = {}
    for r, name in enumerate(names):
        s = scores[name]
        sp, sn = s[pos], s[~pos]
        tx[r] = _compute_midrank(sp)
        ty[r] = _compute_midrank(sn)
        tz[r] = _compute_midrank(np.concatenate([sp, sn]))
        aucs[name] = (tz[r, :m].sum() / (m * n) - (m + 1.0) / (2.0 * n))

    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m

    sx = np.cov(v01)
    sy = np.cov(v10)
    if K == 1:
        sx, sy = np.array([[float(sx)]]), np.array([[float(sy)]])
    cov = sx / m + sy / n
    return aucs, cov


def delong_test(
    y_true: np.ndarray,
    score_a: np.ndarray, score_b: np.ndarray,
) -> Tuple[float, float, float]:
    """Two-sided DeLong p-value for AUROC_A vs AUROC_B on the same
    labels. Returns (auroc_a, auroc_b, p_value)."""
    aucs, cov = delong_roc_variance(y_true, {"a": score_a, "b": score_b})
    diff = aucs["a"] - aucs["b"]
    var  = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return aucs["a"], aucs["b"], 1.0
    z = diff / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return aucs["a"], aucs["b"], float(p)
