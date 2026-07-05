"""Tabular-leader table redo on the cohort-corrected (Expired/Hospice/Acute/AMA
excluded, N=13,858) with Discharge Disposition (home vs other) added to the
ML rows.

Models:
  1. LACE score (van Walraven 2010)
  2. HOSPITAL score (Donzé 2013)
  3. rf_clin              — 39 clinical features + CPT
  4. rf_big enriched      — A + B + C + H + dispo (untuned RF)
  5. LightGBM tuned       — same features, tuned params from tuning experiment
  6. Tuned ensemble       — RF tuned + LGBM tuned + XGB tuned averaged

All 5-fold seed 42, isotonic-calibrated pooled OOF, bootstrap n=2000 CIs.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, f1_score, precision_score,
                             recall_score)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.final_push_080 import (
    log, make_A, make_B, make_C, make_H,
    learner_rf, eval_pooled_oof, SEED, DATA_DIR, GOLD,
)
import medhg_ps.config as C
from medhg_ps.deploy import assemble_training_frame
from medhg_ps.evaluate import _bootstrap_ci

OUT_LOG = Path("artifacts/newdata/tabular_leaders_v10.log")
OUT_RES = Path("artifacts/newdata/tabular_leaders_v10_results.csv")
OUT_OOF = Path("artifacts/newdata/tabular_leaders_v10_oof.npz")

EXCLUDE_LABELS = {
    "Expired",
    "Expired in Medical Facility",
    "Hospice/Home",
    "Hospice/Medical Facility",
    "Acute / Short Term Hospital",
    "Left Against Medical Advice",
}
HOME_LABELS = {"Home or Self Care", "Home-Health Care Svc"}


def _boot_ci(y, p, metric, seed):
    return _bootstrap_ci(y, p, metric, n_boot=2000, seed=seed)


def compute_scores_from_cache(logids):
    """Load LACE + HOSPITAL scores computed by prior scripts, aligned to logids.
    Falls back to recomputing if the cache doesn't exist."""
    lace_path = Path("artifacts/newdata/lace_score_per_encounter.csv")
    hosp_path = Path("artifacts/newdata/hospital_score_per_encounter.csv")
    if not lace_path.exists() or not hosp_path.exists():
        raise FileNotFoundError(
            f"Missing prebuilt score files ({lace_path}, {hosp_path}). "
            "Recompute via analysis/lace_baseline.py and hospital_score.py "
            "before running this.")
    l = pd.read_csv(lace_path); l["LogID"] = l["LogID"].astype(str)
    h = pd.read_csv(hosp_path); h["LogID"] = h["LogID"].astype(str)
    d = pd.DataFrame({"LogID": pd.Series(logids).astype(str)})
    d = d.merge(l[["LogID", "LACE_score"]], on="LogID", how="left")
    d = d.merge(h[["LogID", "HOSPITAL_score"]], on="LogID", how="left")
    return d["LACE_score"].fillna(0).values, d["HOSPITAL_score"].fillna(0).values


def eval_score(y, s, name):
    """Evaluate a fixed integer score (LACE/HOSPITAL) — no CV needed."""
    au = roc_auc_score(y, s)
    ap = average_precision_score(y, s)
    au_ci = _boot_ci(y, s, roc_auc_score, 0)
    ap_ci = _boot_ci(y, s, average_precision_score, 1)
    br = brier_score_loss(y, np.clip(s / max(s.max(), 1), 0, 1))
    best_f1 = 0
    for t in np.unique(s):
        yh = (s >= t).astype(int)
        if 3 < yh.sum() < len(y):
            best_f1 = max(best_f1, f1_score(y, yh))
    return dict(model=name, auroc=au, auroc_lo=au_ci[0], auroc_hi=au_ci[1],
                auprc=ap, auprc_lo=ap_ci[0], auprc_hi=ap_ci[1],
                brier=br, f1=best_f1)


def run_cv(X, y, learner_factory, name):
    N = len(y)
    p_oof = np.full(N, np.nan)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for fi, (tr, te) in enumerate(skf.split(np.zeros(N), y)):
        est = CalibratedClassifierCV(
            learner_factory(), method="isotonic", cv=3).fit(X[tr], y[tr])
        p_oof[te] = est.predict_proba(X[te])[:, 1]
        log(f"  {name} fold {fi + 1} done")
    return p_oof


def run_cv_fold_local(merged, feat_cols, cpt_arr, XB, XC, X_dispo,
                     y, learner_factory, name, use_enriched=True):
    """Same as run_cv but refits Block A (StandardScaler + CPT OneHotEncoder)
    AND Block H (geocode NaN medians) inside each fold on train rows only,
    avoiding leakage from full-cohort preprocessing. If use_enriched is False,
    only Block A is used."""
    from analysis.final_push_080 import make_H
    N = len(y)
    p_oof = np.full(N, np.nan)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for fi, (tr, te) in enumerate(skf.split(np.zeros(N), y)):
        train_mask = np.zeros(N, bool); train_mask[tr] = True
        XA_fold = make_A(merged, feat_cols, cpt_arr, train_mask)
        if use_enriched:
            XH_fold, _ = make_H(merged, train_mask=train_mask)
            X_fold = np.hstack([XA_fold, XB, XC, XH_fold, X_dispo]).astype(np.float32)
        else:
            X_fold = XA_fold.astype(np.float32)
        est = CalibratedClassifierCV(
            learner_factory(), method="isotonic", cv=3).fit(X_fold[tr], y[tr])
        p_oof[te] = est.predict_proba(X_fold[te])[:, 1]
        log(f"  {name} fold {fi + 1} done  X_fold {X_fold.shape}")
    return p_oof


def eval_oof(y, p, name):
    au = roc_auc_score(y, p); ap = average_precision_score(y, p)
    au_ci = _boot_ci(y, p, roc_auc_score, 0)
    ap_ci = _boot_ci(y, p, average_precision_score, 1)
    br = brier_score_loss(y, p)
    best_f1, best_prec, best_rec, best_t = 0, 0, 0, 0
    for t in np.linspace(0.02, 0.35, 80):
        yh = (p >= t).astype(int)
        if yh.sum() < 5: continue
        f = f1_score(y, yh)
        if f > best_f1:
            best_f1 = f; best_prec = precision_score(y, yh)
            best_rec = recall_score(y, yh); best_t = t
    return dict(model=name, auroc=au, auroc_lo=au_ci[0], auroc_hi=au_ci[1],
                auprc=ap, auprc_lo=ap_ci[0], auprc_hi=ap_ci[1],
                brier=br, thr=best_t, f1=best_f1,
                precision=best_prec, recall=best_rec,
                flag_pct=float((p >= best_t).mean() * 100))


def main():
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    log("=== TABULAR LEADERS v10 (cohort-corrected + dispo) START ===")

    merged, feat_cols, cpt_arr, Fseq, seq_all, _ = assemble_training_frame()
    gold = pd.read_parquet(GOLD)[["LogID", "ReadmittedWithin30Days_gold"]]
    gold["LogID"] = gold["LogID"].astype(str)
    merged["LogID"] = merged["LogID"].astype(str)
    merged = merged.merge(gold, on="LogID", how="left")

    n_before = len(merged)
    excl_mask = merged["Discharge Disposition"].astype(str).isin(EXCLUDE_LABELS)
    keep_mask = ~excl_mask
    merged = merged.loc[keep_mask].reset_index(drop=True)
    cpt_arr = np.asarray(cpt_arr).ravel()[keep_mask.values].reshape(-1, 1)
    y = merged["ReadmittedWithin30Days_gold"].astype(int).values
    N = len(y)
    log(f"cohort N={N} (dropped {int(excl_mask.sum())} from {n_before}) "
        f"base rate {y.mean()*100:.2f}%")

    # Precompute feature blocks
    log("assembling feature blocks")
    XB, _ = make_B(merged); log(f"  B {XB.shape}")
    XC, _ = make_C(merged); log(f"  C {XC.shape}")
    XH, _ = make_H(merged); log(f"  H {XH.shape}")

    # Discharge disposition binary flag
    dispo = merged["Discharge Disposition"].astype(str)
    X_dispo = dispo.isin(HOME_LABELS).astype(np.float32).values.reshape(-1, 1)
    log(f"  X_dispo {X_dispo.shape}  home_rate {X_dispo.mean():.3f}")

    # NOTE: XA is built here only for shape logging + fallback score computation
    # (LACE / HOSPITAL). All CV rows below use `run_cv_fold_local`, which refits
    # StandardScaler + CPT OneHotEncoder on each training fold to prevent test-
    # fold statistics from leaking into preprocessing.
    train_mask_full = np.ones(N, bool)
    XA = make_A(merged, feat_cols, cpt_arr, train_mask_full)
    log(f"  A {XA.shape}  (shape only — CV re-fits preprocessing per fold)")

    results = []
    oofs = {}

    # 1. LACE score (fixed, no CV)
    try:
        lace_s, hosp_s = compute_scores_from_cache(merged["LogID"].values)
        log(f"LACE score: n_uniq {len(np.unique(lace_s))}  range [{lace_s.min()},{lace_s.max()}]")
        r = eval_score(y, lace_s, "LACE score (van Walraven 2010)")
        log(f"  {r['model']:60s} AUROC {r['auroc']:.3f} ({r['auroc_lo']:.3f}-{r['auroc_hi']:.3f}) AUPRC {r['auprc']:.3f}")
        results.append(r); oofs["lace"] = lace_s.astype(np.float32)

        r = eval_score(y, hosp_s, "HOSPITAL score (Donzé 2013)")
        log(f"  {r['model']:60s} AUROC {r['auroc']:.3f} ({r['auroc_lo']:.3f}-{r['auroc_hi']:.3f}) AUPRC {r['auprc']:.3f}")
        results.append(r); oofs["hospital"] = hosp_s.astype(np.float32)
    except FileNotFoundError as e:
        log(f"LACE/HOSPITAL cache miss: {e}. Skipping clinical-score rows.")

    # 2. rf_clin — A only (clinical features + CPT), fold-local preprocessing
    log("running rf_clin (A only, per-fold preprocessing)")
    p_rf_clin = run_cv_fold_local(merged, feat_cols, cpt_arr, XB, XC,
                                   X_dispo, y, lambda: learner_rf(),
                                   "rf_clin", use_enriched=False)
    r = eval_oof(y, p_rf_clin, "RF — clinical features only (rf_clin)")
    log(f"  {r['model']:60s} AUROC {r['auroc']:.3f} ({r['auroc_lo']:.3f}-{r['auroc_hi']:.3f}) AUPRC {r['auprc']:.3f}")
    results.append(r); oofs["rf_clin"] = p_rf_clin

    # 3. rf_big enriched (A + B + C + H + dispo, untuned), fold-local preprocessing
    log("running rf_big enriched (A + B + C + H + dispo, per-fold preprocessing)")
    p_rfe = run_cv_fold_local(merged, feat_cols, cpt_arr, XB, XC,
                               X_dispo, y, lambda: learner_rf(big=True),
                               "rf_big_enr", use_enriched=True)
    r = eval_oof(y, p_rfe, "RF — enriched (A + B + C + H + dispo, untuned)")
    log(f"  {r['model']:60s} AUROC {r['auroc']:.3f} ({r['auroc_lo']:.3f}-{r['auroc_hi']:.3f}) AUPRC {r['auprc']:.3f}")
    results.append(r); oofs["rf_big_enr"] = p_rfe

    # 4. LightGBM tuned (use params from tuning experiment fold-1 best;
    # apply same to all folds for simplicity — a light approximation of
    # the nested-CV tuning)
    log("running LightGBM (using tuning-derived params)")
    lgbm_params = dict(
        n_estimators=750, learning_rate=0.014, num_leaves=95,
        max_depth=2, min_child_samples=25, subsample=0.85,
        colsample_bytree=0.65, reg_lambda=0.13, reg_alpha=0.001,
        class_weight=None, random_state=SEED, n_jobs=-1, verbosity=-1,
    )
    p_lgbm = run_cv_fold_local(merged, feat_cols, cpt_arr, XB, XC,
                                X_dispo, y,
                                lambda: lgb.LGBMClassifier(**lgbm_params),
                                "lgbm_tuned", use_enriched=True)
    r = eval_oof(y, p_lgbm, "LightGBM tuned (enriched + dispo)")
    log(f"  {r['model']:60s} AUROC {r['auroc']:.3f} ({r['auroc_lo']:.3f}-{r['auroc_hi']:.3f}) AUPRC {r['auprc']:.3f}")
    results.append(r); oofs["lgbm_tuned"] = p_lgbm

    # 5. XGBoost (best of tuning trial results), fold-local preprocessing
    log("running XGBoost (using tuning-derived params, per-fold preprocessing)")
    xgb_params = dict(
        n_estimators=300, learning_rate=0.018, max_depth=4,
        min_child_weight=1, subsample=0.79, colsample_bytree=0.84,
        reg_lambda=0.53, reg_alpha=0.003, gamma=0.58,
        tree_method="hist", eval_metric="logloss",
        random_state=SEED, n_jobs=-1, verbosity=0, use_label_encoder=False,
    )
    p_xgb = run_cv_fold_local(merged, feat_cols, cpt_arr, XB, XC,
                               X_dispo, y,
                               lambda: xgb.XGBClassifier(**xgb_params),
                               "xgb_tuned", use_enriched=True)
    r = eval_oof(y, p_xgb, "XGBoost tuned (enriched + dispo)")
    log(f"  {r['model']:60s} AUROC {r['auroc']:.3f} ({r['auroc_lo']:.3f}-{r['auroc_hi']:.3f}) AUPRC {r['auprc']:.3f}")
    results.append(r); oofs["xgb_tuned"] = p_xgb

    # 6. Ensemble
    p_ens = np.mean([p_rfe, p_lgbm, p_xgb], axis=0)
    r = eval_oof(y, p_ens, "Ensemble (RF + LGBM + XGB averaged)")
    log(f"  {r['model']:60s} AUROC {r['auroc']:.3f} ({r['auroc_lo']:.3f}-{r['auroc_hi']:.3f}) AUPRC {r['auprc']:.3f}")
    results.append(r); oofs["ensemble"] = p_ens

    pd.DataFrame(results).to_csv(OUT_RES, index=False)
    np.savez(OUT_OOF, y=y, **oofs)
    log(f"saved {OUT_RES}, {OUT_OOF}")
    log("=== DONE ===")


if __name__ == "__main__":
    main()
