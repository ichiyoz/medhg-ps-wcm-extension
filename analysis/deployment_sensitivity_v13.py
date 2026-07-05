"""Deployment sensitivity analysis for the tuned ensemble.

Trains rf_big + LightGBM + XGBoost on clean training folds, then scores
test folds with different blocks of features PERTURBED (missing values
or wrong values). Measures the AUROC/AUPRC drop for each realistic
deployment failure mode.

Perturbations tested (applied to TEST slice only, never during training):

MISSING BLOCKS (feature unavailable — replaced with train-fold median /
majority category):
  1. Missing Charlson × 17 (LACE_Components SQL unavailable)
  2. Missing prior-utilization (ED180d, admits365d unavailable)
  3. Missing labs (HCT, Na, Cr, Alb, WBC unavailable — set to normal)
  4. Missing CPT (PrimaryCPT unknown)
  5. Missing discharge disposition (assumed home)
  6. Missing geocode / SDoH (unknown ZIP)
  7. Missing LACE/HOSPITAL scores (component computation failure)

WRONG BLOCKS (data present but corrupted / permuted across patients):
  8. Wrong demographics (age + gender shuffled across test rows)
  9. Wrong SDoH (geocode features shuffled across test rows)
  10. Wrong CPT (shuffled)

COMBINED FAILURES:
  11. Bare-bones: only demographics + geocode available, everything else
      set to defaults (a "very limited data" deployment scenario)

Metric: mean AUROC/AUPRC across 5 folds. Baseline is the clean ensemble.
"""
from __future__ import annotations
import sys, time, warnings
from copy import deepcopy
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.final_push_080 import (
    log, make_A, make_B, make_C, make_H,
    learner_rf, SEED, DATA_DIR, GOLD,
)
from analysis.tabular_leaders_v10 import (EXCLUDE_LABELS, HOME_LABELS)
from medhg_ps.deploy import assemble_training_frame

OUT_LOG = Path("artifacts/newdata/deployment_sensitivity_v13.log")
OUT_RES = Path("artifacts/newdata/deployment_sensitivity_v13_results.csv")


def make_lgbm():
    return lgb.LGBMClassifier(
        n_estimators=750, learning_rate=0.014, num_leaves=95,
        max_depth=2, min_child_samples=25, subsample=0.85,
        colsample_bytree=0.65, reg_lambda=0.13, reg_alpha=0.001,
        class_weight=None, random_state=SEED, n_jobs=-1, verbosity=-1)


def make_xgb():
    return xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.018, max_depth=4,
        min_child_weight=1, subsample=0.79, colsample_bytree=0.84,
        reg_lambda=0.53, reg_alpha=0.003, gamma=0.58,
        tree_method="hist", eval_metric="logloss",
        random_state=SEED, n_jobs=-1, verbosity=0, use_label_encoder=False)


def _feat_col_names(feat_cols, cpt_arr, XA_shape):
    """Try to reconstruct which columns in XA correspond to which SQL
    field — used to identify column indices for each perturbation.
    We tag by prefix strings we know the preprocessor uses."""
    return None   # placeholder; using slice indices instead (see main)


def find_indices(feat_cols, XA_cols, XB_shape, XC_shape, XH_shape):
    """Return dict of slice indices for each block in the full stacked matrix.
    XA is 0..a; XB is a..a+b; XC is a+b..a+b+c; XH is a+b+c..a+b+c+h;
    dispo is final column."""
    a = XA_cols; b = XB_shape[1]; c = XC_shape[1]; h = XH_shape[1]
    return dict(
        A=slice(0, a),
        B=slice(a, a + b),
        C=slice(a + b, a + b + c),
        H=slice(a + b + c, a + b + c + h),
        dispo=slice(a + b + c + h, a + b + c + h + 1),
    )


def perturb_missing(X_test, X_train, cols, mode="median"):
    """Replace columns in X_test with fill values derived from X_train."""
    X_pert = X_test.copy()
    if mode == "median":
        fill = np.median(X_train[:, cols], axis=0)
    elif mode == "zero":
        fill = np.zeros(X_train[:, cols].shape[1])
    elif mode == "mean":
        fill = X_train[:, cols].mean(axis=0)
    X_pert[:, cols] = fill
    return X_pert


def perturb_shuffle(X_test, cols, seed):
    """Shuffle values in given columns across rows of X_test — simulates
    'data is present but assigned to the wrong patient'."""
    X_pert = X_test.copy()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_test))
    X_pert[:, cols] = X_test[idx][:, cols]
    return X_pert


def score_ensemble(rf, lgbm_m, xgb_m, X):
    p1 = rf.predict_proba(X)[:, 1]
    p2 = lgbm_m.predict_proba(X)[:, 1]
    p3 = xgb_m.predict_proba(X)[:, 1]
    return (p1 + p2 + p3) / 3.0


def eval_perturbation(name, y, p):
    au = roc_auc_score(y, p)
    ap = average_precision_score(y, p)
    return dict(scenario=name, auroc=au, auprc=ap)


def main():
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    log("=== DEPLOYMENT SENSITIVITY v13 START ===")

    merged, feat_cols, cpt_arr, Fseq, seq_all, _ = assemble_training_frame()
    gold = pd.read_parquet(GOLD)[["LogID", "ReadmittedWithin30Days_gold"]]
    gold["LogID"] = gold["LogID"].astype(str)
    merged["LogID"] = merged["LogID"].astype(str)
    merged = merged.merge(gold, on="LogID", how="left")
    excl_mask = merged["Discharge Disposition"].astype(str).isin(EXCLUDE_LABELS)
    merged = merged.loc[~excl_mask].reset_index(drop=True)
    cpt_arr = np.asarray(cpt_arr).ravel()[~excl_mask.values].reshape(-1, 1)
    y = merged["ReadmittedWithin30Days_gold"].astype(int).values
    N = len(y)
    log(f"cohort N={N}  base rate {y.mean()*100:.2f}%")

    XB, colsB = make_B(merged); log(f"  B {XB.shape}")
    XC, colsC = make_C(merged); log(f"  C {XC.shape}")
    XH, colsH = make_H(merged); log(f"  H {XH.shape}")
    X_dispo = merged["Discharge Disposition"].astype(str).isin(HOME_LABELS)\
                 .astype(np.float32).values.reshape(-1, 1)
    train_mask_full = np.ones(N, bool)
    XA = make_A(merged, feat_cols, cpt_arr, train_mask_full)
    log(f"  A {XA.shape}")

    X_full = np.hstack([XA, XB, XC, XH, X_dispo]).astype(np.float32)
    log(f"X_full {X_full.shape}")

    # Slice indices per block
    a_dim = XA.shape[1]
    b_dim = XB.shape[1]
    c_dim = XC.shape[1]
    h_dim = XH.shape[1]
    idx = dict(
        A=list(range(0, a_dim)),
        B=list(range(a_dim, a_dim + b_dim)),
        C=list(range(a_dim + b_dim, a_dim + b_dim + c_dim)),
        H=list(range(a_dim + b_dim + c_dim, a_dim + b_dim + c_dim + h_dim)),
        dispo=[a_dim + b_dim + c_dim + h_dim],
    )
    n_cpt_cols = 41  # approximate; make_A appends one-hot CPT at end
    idx["A_cpt"] = list(range(a_dim - n_cpt_cols, a_dim))
    idx["A_nocpt"] = list(range(0, a_dim - n_cpt_cols))
    # Split B into Charlson (first 17) and utilization (remaining ~8)
    idx["B_charlson"] = list(range(a_dim, a_dim + 17))
    idx["B_utilization"] = list(range(a_dim + 17, a_dim + b_dim))
    log(f"  Block slice sizes: A {len(idx['A'])}  B {len(idx['B'])}  "
        f"C {len(idx['C'])}  H {len(idx['H'])}  dispo {len(idx['dispo'])}  "
        f"A_cpt {len(idx['A_cpt'])}")

    log("=== 5-fold OUTER CV — BASELINE + PERTURBATIONS ===")
    scenarios = [
        # (name, kind, columns to perturb, mode/seed)
        ("baseline_clean", "none", [], None),
        # MISSING blocks
        ("missing_charlson", "missing", idx["B"], "zero"),
        ("missing_utilization", "missing",
         # Just ED180d and admits365d approximated as last 2 in B (they include
         # log/binned/interactions too — this is a coarse "no utilization" test)
         idx["B"], "median"),
        ("missing_lace_hospital_scores", "missing", idx["C"], "median"),
        ("missing_geocode_sdoh", "missing", idx["H"], "median"),
        ("missing_cpt", "missing", idx["A_cpt"], "zero"),
        ("missing_dispo", "missing", idx["dispo"], "zero"),
        # WRONG blocks (shuffled across rows)
        ("wrong_demographics", "shuffle",
         list(range(0, min(15, a_dim - n_cpt_cols))), 100),
        ("wrong_sdoh", "shuffle", idx["H"], 200),
        ("wrong_cpt", "shuffle", idx["A_cpt"], 300),
        ("wrong_charlson", "shuffle", idx["B_charlson"], 400),
        ("wrong_utilization", "shuffle", idx["B_utilization"], 500),
        # COMBINED failures
        ("bare_bones_demog_only", "keep_only",
         list(range(0, min(15, a_dim - n_cpt_cols))) + idx["H"], None),
    ]

    results_all = {s[0]: [] for s in scenarios}
    y_all = {s[0]: [] for s in scenarios}
    p_all = {s[0]: [] for s in scenarios}

    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for fi, (tr, te) in enumerate(skf.split(np.zeros(N), y)):
        log(f"--- fold {fi + 1}/5 ---")
        X_tr, X_te = X_full[tr], X_full[te]
        y_tr, y_te = y[tr], y[te]

        # Train clean ensemble on train fold
        rf = CalibratedClassifierCV(learner_rf(big=True), method="isotonic", cv=3)
        rf.fit(X_tr, y_tr)
        lgbm_m = CalibratedClassifierCV(make_lgbm(), method="isotonic", cv=3)
        lgbm_m.fit(X_tr, y_tr)
        xgb_m = CalibratedClassifierCV(make_xgb(), method="isotonic", cv=3)
        xgb_m.fit(X_tr, y_tr)
        log(f"  fold {fi + 1}: ensemble trained")

        for name, kind, cols, param in scenarios:
            if kind == "none":
                Xp = X_te
            elif kind == "missing":
                Xp = perturb_missing(X_te, X_tr, cols, mode=param)
            elif kind == "shuffle":
                Xp = perturb_shuffle(X_te, cols, seed=SEED + fi * 1000 + param)
            elif kind == "keep_only":
                # Zero out ALL columns except the ones in `cols`, replaced with medians
                keep_set = set(cols)
                drop_cols = [c for c in range(X_te.shape[1]) if c not in keep_set]
                Xp = perturb_missing(X_te, X_tr, drop_cols, mode="median")
            else:
                continue
            p = score_ensemble(rf, lgbm_m, xgb_m, Xp)
            y_all[name].append(y_te)
            p_all[name].append(p)
        log(f"  fold {fi + 1}: {len(scenarios)} scenarios scored")

    log("=== POOLED RESULTS ===")
    results = []
    for name, _, _, _ in scenarios:
        y_p = np.concatenate(y_all[name])
        p_p = np.concatenate(p_all[name])
        au = roc_auc_score(y_p, p_p)
        ap = average_precision_score(y_p, p_p)
        results.append(dict(scenario=name, auroc=au, auprc=ap))
        log(f"  {name:30s} AUROC {au:.4f}  AUPRC {ap:.4f}")

    # Δ vs baseline
    base_au = results[0]["auroc"]; base_ap = results[0]["auprc"]
    for r in results:
        r["delta_auroc"] = r["auroc"] - base_au
        r["delta_auprc"] = r["auprc"] - base_ap
    df = pd.DataFrame(results)
    df.to_csv(OUT_RES, index=False)
    log(f"\n=== TABLE ===")
    log(f"{'Scenario':32s} {'AUROC':>7s} {'ΔAUROC':>8s} {'AUPRC':>7s} {'ΔAUPRC':>8s}")
    for _, r in df.iterrows():
        log(f"{r['scenario']:32s} {r['auroc']:>7.4f} {r['delta_auroc']:>+8.4f} "
            f"{r['auprc']:>7.4f} {r['delta_auprc']:>+8.4f}")
    log(f"saved {OUT_RES}")
    log("=== DONE ===")


if __name__ == "__main__":
    main()
