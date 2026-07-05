"""Deployment sensitivity analysis for the Flow model (v9b).

Same setup as v13 but on the Flow model:
  - Lean tabular (Age, Gender, Charlson x 17, ED180d, admits365d, geocode)
  - Multi-attr flow GRU embedding (64-d)
  - Static graph context: CPT one-hot (41) + provider role mix (5)
    + seq scalars (2: n_orders, bundle_fraction) + dispo (1)

Perturbations applied only to test rows (train once per fold on clean data):

MISSING blocks (feature unavailable — replaced with train median or zero):
  1. Missing Charlson × 17 (LACE_Components SQL down)
  2. Missing prior utilization (ED180d, admits365d)
  3. Missing geocode / SDoH
  4. Missing CPT (top-40 one-hot)
  5. Missing dispo
  6. Missing order sequence (Order_sequence pipeline down) — GRU embedding zeroed
  7. Missing provider role mix (A2/A4 pipeline down)
  8. Missing sequence scalars

WRONG blocks (data present but wrong patient):
  9. Wrong demographics (age + gender shuffled)
  10. Wrong SDoH (geocode shuffled)
  11. Wrong CPT (shuffled)
  12. Wrong order sequence (GRU embedding shuffled across rows)
  13. Wrong provider role mix (shuffled)

COMBINED failures:
  14. Bare-bones: only demographics + geocode kept

Metric: pooled AUROC/AUPRC over 5-fold OOF. Baseline = clean Flow model.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.final_push_080 import (log, learner_rf, SEED, DATA_DIR, GOLD)
from analysis.final_push_080_v9b import (
    make_tabular_lean, make_static_context, build_sequences,
    train_gru_and_encode, GRU_HIDDEN,
)
HOME_LABELS = {"Home or Self Care", "Home-Health Care Svc"}

from medhg_ps.data import load_raw, load_order_sequence, collapse_order_runs
from medhg_ps.deploy import assemble_training_frame

EXCLUDE_LABELS = {
    "Expired", "Expired in Medical Facility",
    "Hospice/Home", "Hospice/Medical Facility",
    "Acute / Short Term Hospital", "Left Against Medical Advice",
}

OUT_LOG = Path("artifacts/newdata/deployment_sensitivity_flow_v13b.log")
OUT_RES = Path("artifacts/newdata/deployment_sensitivity_flow_v13b_results.csv")


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


def perturb_missing(X_test, X_train, cols, mode="median"):
    X_pert = X_test.copy()
    if mode == "median":
        fill = np.median(X_train[:, cols], axis=0)
    elif mode == "zero":
        fill = np.zeros(X_train[:, cols].shape[1])
    X_pert[:, cols] = fill
    return X_pert


def perturb_shuffle(X_test, cols, seed):
    X_pert = X_test.copy()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_test))
    X_pert[:, cols] = X_test[idx][:, cols]
    return X_pert


def score_ensemble(rf, lgbm_m, xgb_m, X):
    return (rf.predict_proba(X)[:, 1]
            + lgbm_m.predict_proba(X)[:, 1]
            + xgb_m.predict_proba(X)[:, 1]) / 3.0


def main():
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    log("=== DEPLOYMENT SENSITIVITY — FLOW MODEL v13b START ===")

    merged, feat_cols, cpt_arr, Fseq, seq_all, _ = assemble_training_frame()
    gold = pd.read_parquet(GOLD)[["LogID", "ReadmittedWithin30Days_gold"]]
    gold["LogID"] = gold["LogID"].astype(str)
    merged["LogID"] = merged["LogID"].astype(str)
    merged = merged.merge(gold, on="LogID", how="left")
    excl = merged["Discharge Disposition"].astype(str).isin(EXCLUDE_LABELS)
    merged = merged.loc[~excl].reset_index(drop=True)
    cpt_arr = np.asarray(cpt_arr).ravel()[~excl.values].reshape(-1, 1)
    y = merged["ReadmittedWithin30Days_gold"].astype(int).values
    N = len(y)
    log(f"cohort N={N}  base rate {y.mean()*100:.2f}%")

    # Build sequences ONCE
    log("building order sequences")
    orders_df = collapse_order_runs(load_order_sequence())
    raw = load_raw()
    all_logids = merged["LogID"].astype(str).values
    lid2idx = {lid: i for i, lid in enumerate(all_logids)}
    order_seqs, time_seqs, src_seqs, lens, n_ord, n_src = \
        build_sequences(orders_df, lid2idx, N)
    log(f"  sequences built, median len {int(np.median(lens))}")

    log("=== 5-fold OUTER CV ===")
    scenarios = [
        # (name, kind, target_block, param)
        ("baseline_clean", "none", None, None),
        # MISSING
        ("missing_charlson", "missing", "charlson", "zero"),
        ("missing_utilization", "missing", "utilization", "median"),
        ("missing_geocode_sdoh", "missing", "geocode", "median"),
        ("missing_cpt", "missing", "cpt", "zero"),
        ("missing_dispo", "missing", "dispo", "zero"),
        ("missing_order_sequence", "missing", "gru", "zero"),
        ("missing_provider_roles", "missing", "role", "zero"),
        ("missing_seq_scalars", "missing", "seq_scalars", "median"),
        # WRONG
        ("wrong_demographics", "shuffle", "demographics", 100),
        ("wrong_sdoh", "shuffle", "geocode", 200),
        ("wrong_cpt", "shuffle", "cpt", 300),
        ("wrong_order_sequence", "shuffle", "gru", 400),
        ("wrong_provider_roles", "shuffle", "role", 500),
        ("wrong_charlson", "shuffle", "charlson", 600),
        ("wrong_utilization", "shuffle", "utilization", 700),
        # COMBINED
        ("bare_bones_demog_only", "keep_only", "demographics+geocode", None),
    ]

    y_all = {s[0]: [] for s in scenarios}
    p_all = {s[0]: [] for s in scenarios}

    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for fi, (tr, te) in enumerate(skf.split(np.zeros(N), y)):
        log(f"--- fold {fi + 1}/5 ---")
        train_mask = np.zeros(N, bool); train_mask[tr] = True

        # Build blocks (train-fold-fit preprocessing)
        X_tab = make_tabular_lean(merged, train_mask)
        ctx = make_static_context(merged, cpt_arr, orders_df,
                                   raw.enc_prov_edges, raw.prov_attrs, train_mask)
        X_cpt = ctx["X_cpt"]; X_role = ctx["X_role"]
        X_seq_scalar = ctx["X_seq"]; X_dispo = ctx["X_dispo"]
        # Train GRU on train fold, encode ALL rows
        X_gru = train_gru_and_encode(
            order_seqs, time_seqs, src_seqs, lens, y, tr,
            n_ord, n_src, epochs=5, seed=SEED + fi)

        # Compose full feature matrix
        X_full = np.hstack([X_tab, X_gru, X_cpt, X_role,
                            X_seq_scalar, X_dispo]).astype(np.float32)
        log(f"  X_full {X_full.shape}")

        # Block slices in the concatenated matrix
        a = X_tab.shape[1]                          # 29 columns
        b = X_gru.shape[1]                          # 64
        c = X_cpt.shape[1]                          # 41
        d = X_role.shape[1]                         # 5
        e = X_seq_scalar.shape[1]                   # 2
        # tabular sub-blocks: demographics (0:3), charlson (3:20),
        # utilization (20:22), geocode (22:28) — depends on make_tabular_lean order
        demo_end = 3       # AgeYears + Gender one-hots (~3)
        cci_end = demo_end + 17   # 3..20
        util_end = cci_end + 2    # 20..22
        geo_end = util_end + 6    # 22..28
        idx = dict(
            demographics=list(range(0, demo_end)),
            charlson=list(range(demo_end, cci_end)),
            utilization=list(range(cci_end, util_end)),
            geocode=list(range(util_end, geo_end)),
            gru=list(range(a, a + b)),
            cpt=list(range(a + b, a + b + c)),
            role=list(range(a + b + c, a + b + c + d)),
            seq_scalars=list(range(a + b + c + d, a + b + c + d + e)),
            dispo=[a + b + c + d + e],
        )

        # Train ensemble
        X_tr, X_te = X_full[tr], X_full[te]
        y_tr, y_te = y[tr], y[te]
        rf = CalibratedClassifierCV(learner_rf(big=True), method="isotonic", cv=3)
        rf.fit(X_tr, y_tr)
        lgbm_m = CalibratedClassifierCV(make_lgbm(), method="isotonic", cv=3)
        lgbm_m.fit(X_tr, y_tr)
        xgb_m = CalibratedClassifierCV(make_xgb(), method="isotonic", cv=3)
        xgb_m.fit(X_tr, y_tr)
        log(f"  fold {fi + 1}: ensemble trained")

        # Score each perturbation
        for name, kind, block, param in scenarios:
            if kind == "none":
                Xp = X_te
            elif kind == "missing":
                cols = idx[block]
                Xp = perturb_missing(X_te, X_tr, cols, mode=param)
            elif kind == "shuffle":
                cols = idx[block]
                Xp = perturb_shuffle(X_te, cols, seed=SEED + fi * 1000 + param)
            elif kind == "keep_only":
                keep = set(idx["demographics"]) | set(idx["geocode"])
                drop = [c for c in range(X_te.shape[1]) if c not in keep]
                Xp = perturb_missing(X_te, X_tr, drop, mode="median")
            else:
                continue
            p = score_ensemble(rf, lgbm_m, xgb_m, Xp)
            y_all[name].append(y_te); p_all[name].append(p)

    # Aggregate
    log("=== POOLED RESULTS ===")
    results = []
    for name, _, _, _ in scenarios:
        y_p = np.concatenate(y_all[name])
        p_p = np.concatenate(p_all[name])
        au = roc_auc_score(y_p, p_p)
        ap = average_precision_score(y_p, p_p)
        results.append(dict(scenario=name, auroc=au, auprc=ap))
        log(f"  {name:30s} AUROC {au:.4f}  AUPRC {ap:.4f}")

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
