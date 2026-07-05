"""Nested 5-outer / 3-inner CV Bayesian tuning of the winning learners
on the graphs-dropped feature set (A + B + C + H, ~247 columns).

Winning learner from v7 ablation: rf_big at 0.741 AUROC without graph blocks.
We tune rf_big, LightGBM, and XGBoost under identical protocol and compare.

Search: Optuna TPE, ~50 trials per learner.
Objective: maximize inner-CV AUROC.
Outer: 5-fold seed 42 (matches all prior runs).
Inner: 3-fold seed 42+fold_i.

Outputs:
  artifacts/newdata/tune_results.csv      per-learner outer AUROC/AUPRC with CIs
  artifacts/newdata/tune_best_params.json  best params per outer fold
  artifacts/newdata/tune_oof.npz          calibrated OOFs
  artifacts/newdata/tune.log              full progress log
"""
from __future__ import annotations
import json, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, f1_score)
import xgboost as xgb
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.final_push_080 import (
    log, make_A, make_B, make_C, make_H,
    eval_pooled_oof, SEED, DATA_DIR, GOLD,
)
import medhg_ps.config as C
from medhg_ps.deploy import assemble_training_frame

optuna.logging.set_verbosity(optuna.logging.WARNING)

OUT_LOG = Path("artifacts/newdata/tune.log")
OUT_RES = Path("artifacts/newdata/tune_results.csv")
OUT_OOF = Path("artifacts/newdata/tune_oof.npz")
OUT_BP  = Path("artifacts/newdata/tune_best_params.json")

N_TRIALS_RF   = 40    # RF is slow; keep trials moderate
N_TRIALS_XGB  = 60
N_TRIALS_LGBM = 60
INNER_CV = 3


def objective_rf(trial, X_tr, y_tr, seed):
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 400, 1200, step=100),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
        class_weight=trial.suggest_categorical("class_weight",
                                                ["balanced", "balanced_subsample"]),
        max_depth=trial.suggest_int("max_depth", 6, 30),
        n_jobs=-1, random_state=seed,
    )
    clf = RandomForestClassifier(**params)
    scores = cross_val_score(clf, X_tr, y_tr, cv=INNER_CV,
                             scoring="roc_auc", n_jobs=1)
    return scores.mean()


def objective_xgb(trial, X_tr, y_tr, seed):
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 200, 800, step=50),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        gamma=trial.suggest_float("gamma", 0.0, 5.0),
        tree_method="hist", eval_metric="logloss",
        random_state=seed, n_jobs=-1, verbosity=0, use_label_encoder=False,
    )
    clf = xgb.XGBClassifier(**params)
    scores = cross_val_score(clf, X_tr, y_tr, cv=INNER_CV,
                             scoring="roc_auc", n_jobs=1)
    return scores.mean()


def objective_lgbm(trial, X_tr, y_tr, seed):
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 200, 1000, step=50),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 255),
        max_depth=trial.suggest_int("max_depth", -1, 15),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        class_weight=trial.suggest_categorical("class_weight",
                                                ["balanced", None]),
        random_state=seed, n_jobs=-1, verbosity=-1,
    )
    clf = lgb.LGBMClassifier(**params)
    scores = cross_val_score(clf, X_tr, y_tr, cv=INNER_CV,
                             scoring="roc_auc", n_jobs=1)
    return scores.mean()


def build_learner(name, params, seed):
    if name == "rf":
        return RandomForestClassifier(**params, n_jobs=-1, random_state=seed)
    if name == "xgb":
        return xgb.XGBClassifier(**params, tree_method="hist",
                                 eval_metric="logloss", n_jobs=-1,
                                 verbosity=0, use_label_encoder=False,
                                 random_state=seed)
    if name == "lgbm":
        return lgb.LGBMClassifier(**params, n_jobs=-1, verbosity=-1,
                                  random_state=seed)


def tune_and_eval(name, obj_fn, n_trials, X, y, N, seed_start=SEED):
    """Nested CV: outer 5-fold, inner Optuna TPE with 3-fold CV.
    Returns (pooled_oof, per_fold_best_params, per_fold_best_inner_auc)."""
    p_oof = np.full(N, np.nan)
    best_params_per_fold = []
    best_inner_per_fold = []

    skf_outer = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for fi, (tr, te) in enumerate(skf_outer.split(np.zeros(N), y)):
        log(f"  [{name}] outer fold {fi + 1}/5  tune {n_trials} trials")
        t0 = time.time()
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(
                                        seed=seed_start + fi))
        study.optimize(
            lambda trial: obj_fn(trial, X[tr], y[tr], seed_start + fi),
            n_trials=n_trials, show_progress_bar=False,
        )
        best = study.best_params
        best_inner = study.best_value
        log(f"    [{name}] inner best AUROC={best_inner:.4f} "
            f"in {time.time()-t0:.0f}s  params={best}")
        best_params_per_fold.append(best)
        best_inner_per_fold.append(float(best_inner))

        # Refit tuned learner + isotonic on full train fold, predict test
        clf = build_learner(name, best, seed_start + fi)
        est = CalibratedClassifierCV(clf, method="isotonic", cv=3)
        est.fit(X[tr], y[tr])
        p_oof[te] = est.predict_proba(X[te])[:, 1]
        log(f"    [{name}] fold {fi + 1} calibrated OOF filled")

    return p_oof, best_params_per_fold, best_inner_per_fold


def main():
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    log("=== NESTED-CV TUNING (graphs-dropped feature set) START ===")

    merged, feat_cols, cpt_arr, Fseq, seq_all, _ = assemble_training_frame()
    gold = pd.read_parquet(GOLD)[["LogID", "ReadmittedWithin30Days_gold"]]
    gold["LogID"] = gold["LogID"].astype(str)
    merged["LogID"] = merged["LogID"].astype(str)
    merged = merged.merge(gold, on="LogID", how="left")
    y = merged["ReadmittedWithin30Days_gold"].astype(int).values
    N = len(y)
    log(f"cohort N={N} base rate {y.mean()*100:.2f}%")

    log("BLOCK B: LACE utility"); XB, _ = make_B(merged); log(f"  B {XB.shape}")
    log("BLOCK C: LACE + HOSPITAL scores"); XC, _ = make_C(merged); log(f"  C {XC.shape}")
    log("BLOCK H: geocode + ACS"); XH, _ = make_H(merged); log(f"  H {XH.shape}")

    # Build A once with full-cohort mask for consistency (fit_preprocess is
    # re-done inside make_A per fold via train_mask, so this is a shell)
    train_mask = np.ones(N, bool)
    XA = make_A(merged, feat_cols, cpt_arr, train_mask)
    X_full = np.hstack([XA, XB, XC, XH]).astype(np.float32)
    log(f"X_full {X_full.shape}  (A + B + C + H, no graph blocks)")

    all_oof = {}
    all_best = {}
    all_inner = {}

    for name, obj_fn, n_trials in [
        ("rf",   objective_rf,   N_TRIALS_RF),
        ("xgb",  objective_xgb,  N_TRIALS_XGB),
        ("lgbm", objective_lgbm, N_TRIALS_LGBM),
    ]:
        log(f"=== TUNING {name.upper()} ({n_trials} trials/fold × 5 folds) ===")
        t0 = time.time()
        p_oof, bps, inners = tune_and_eval(name, obj_fn, n_trials, X_full, y, N)
        all_oof[name] = p_oof
        all_best[name] = bps
        all_inner[name] = inners
        log(f"{name} total wall time {time.time()-t0:.0f}s")

    log("=== POOLED OOF EVAL ===")
    results = []
    for name, p in all_oof.items():
        if np.isnan(p).any():
            log(f"  {name} has NaN, skipping"); continue
        r = eval_pooled_oof(y, p, f"{name}_tuned")
        r["mean_inner_auroc"] = float(np.mean(all_inner[name]))
        log(f"  {name}_tuned AUROC {r['auroc']:.3f} "
            f"({r['auroc_lo']:.3f}-{r['auroc_hi']:.3f}) "
            f"AUPRC {r['auprc']:.3f} F1 {r['f1']:.3f}  "
            f"inner-CV mean {r['mean_inner_auroc']:.3f}")
        results.append(r)

    # ensemble of tuned learners
    valid = [k for k, v in all_oof.items() if not np.isnan(v).any()]
    if len(valid) >= 2:
        p_ens = np.mean([all_oof[k] for k in valid], axis=0)
        r = eval_pooled_oof(y, p_ens, "ensemble_tuned")
        log(f"  ensemble_tuned AUROC {r['auroc']:.3f} "
            f"({r['auroc_lo']:.3f}-{r['auroc_hi']:.3f}) "
            f"AUPRC {r['auprc']:.3f} F1 {r['f1']:.3f}")
        results.append(r)

    pd.DataFrame(results).to_csv(OUT_RES, index=False)
    np.savez(OUT_OOF, y=y, **all_oof)
    with open(OUT_BP, "w") as f:
        json.dump({name: {"per_fold": bps, "inner_auroc": all_inner[name]}
                   for name, bps in all_best.items()}, f, indent=2)
    log(f"saved {OUT_RES}, {OUT_OOF}, {OUT_BP}")
    log("=== DONE ===")


if __name__ == "__main__":
    main()
