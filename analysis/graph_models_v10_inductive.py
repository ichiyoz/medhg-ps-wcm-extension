"""Fold-honest INDUCTIVE evaluation of the graph rows in Table 3.

For each fold: build the graph on TRAIN encounters only, train the GNN
or Node2Vec, then embed test encounters by aggregating neighbours'
trained embeddings (GraphSAGE-style inductive step). Test encounters
never participate in training message passing / walks — this mirrors
what would happen at deployment on a genuinely new patient.

Rows produced (Table 3 update):
  MedHG-PS (original, A3 care-unit graph)  — inductive ie-HGCN
  MedHG-PS (orders as A3)                  — inductive ie-HGCN with
                                             order-groups substituted
                                             for units
  Node2Vec + trees                         — inductive Node2Vec

Downstream: same tuned tree ensemble (rf_big + LightGBM + XGBoost) with
per-fold Block A + Block H fold-local preprocessing, matching the
tabular_leaders_v10 protocol.

Save inductive results to:
  artifacts/newdata/graph_models_v10_inductive_results.csv
  artifacts/newdata/graph_models_v10_inductive_oof.npz
"""
from __future__ import annotations
import sys, time, warnings
from copy import copy as shallow_copy
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, f1_score, precision_score,
                             recall_score)
import xgboost as xgb
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.final_push_080 import (
    log, make_A, make_B, make_C, make_H,
    learner_rf, SEED, DATA_DIR, GOLD,
)
from analysis.final_push_080_v2 import GNN_CFG, GNN_DEV
from analysis.final_push_080_v3 import build_order_group_substrate
from analysis.final_push_080_v6 import precompute_graph
from analysis.final_push_080_v8 import _norm_id
from analysis.inductive_graph_helpers import (
    train_gnn_and_encode_inductive, compute_node2vec_inductive,
)

from medhg_ps.data import (load_raw, load_order_sequence, collapse_order_runs,
                            build_provider_features, build_unit_features)
from medhg_ps.deploy import assemble_training_frame
from medhg_ps.evaluate import _bootstrap_ci

OUT_LOG = Path("artifacts/newdata/graph_models_v10_inductive.log")
OUT_RES = Path("artifacts/newdata/graph_models_v10_inductive_results.csv")
OUT_OOF = Path("artifacts/newdata/graph_models_v10_inductive_oof.npz")

EXCLUDE_LABELS = {"Expired","Expired in Medical Facility","Hospice/Home",
                  "Hospice/Medical Facility","Acute / Short Term Hospital",
                  "Left Against Medical Advice"}
HOME_LABELS = {"Home or Self Care","Home-Health Care Svc"}


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


def _boot(y, p, metric, seed):
    return _bootstrap_ci(y, p, metric, n_boot=2000, seed=seed)


def _eval(y, p, name):
    au = roc_auc_score(y, p); ap = average_precision_score(y, p)
    au_ci = _boot(y, p, roc_auc_score, 0)
    ap_ci = _boot(y, p, average_precision_score, 1)
    br = brier_score_loss(y, p)
    best_f1 = 0
    for t in np.linspace(0.02, 0.35, 80):
        yh = (p >= t).astype(int)
        if yh.sum() < 5: continue
        best_f1 = max(best_f1, f1_score(y, yh))
    return dict(model=name, auroc=au, auroc_lo=au_ci[0], auroc_hi=au_ci[1],
                auprc=ap, auprc_lo=ap_ci[0], auprc_hi=ap_ci[1],
                brier=br, f1=best_f1)


def run_ensemble_fold_local(merged, feat_cols, cpt_arr, XB, XC, X_dispo,
                             y, name, get_extra_features_fn):
    """5-fold CV with rf_big + LGBM + XGB downstream, mean = ensemble.
    Blocks A + H refit inside each fold (train-only preprocessing).
    get_extra_features_fn(tr) returns an (N, D) array — must be inductive
    (test-fold rows filled by aggregating training-only neighbours)."""
    N = len(y)
    p_rf = np.full(N, np.nan)
    p_lgbm = np.full(N, np.nan)
    p_xgb = np.full(N, np.nan)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for fi, (tr, te) in enumerate(skf.split(np.zeros(N), y)):
        train_mask = np.zeros(N, bool); train_mask[tr] = True
        XA_fold = make_A(merged, feat_cols, cpt_arr, train_mask)
        XH_fold, _ = make_H(merged, train_mask=train_mask)
        X_base = np.hstack([XA_fold, XB, XC, XH_fold, X_dispo]).astype(np.float32)
        log(f"  fold {fi + 1}/5: computing INDUCTIVE graph features")
        sys.stdout.flush()
        t0 = time.time()
        X_extra = get_extra_features_fn(tr)
        log(f"  fold {fi + 1}/5: inductive features done in {time.time()-t0:.0f}s")
        sys.stdout.flush()
        X = np.hstack([X_base, X_extra]).astype(np.float32)
        log(f"  fold {fi + 1}/5: X {X.shape}")
        sys.stdout.flush()
        for lname, mk, arr in [("rf_big", lambda: learner_rf(big=True), p_rf),
                                ("lgbm", make_lgbm, p_lgbm),
                                ("xgb", make_xgb, p_xgb)]:
            try:
                est = CalibratedClassifierCV(mk(), method="isotonic",
                                             cv=3).fit(X[tr], y[tr])
                arr[te] = est.predict_proba(X[te])[:, 1]
            except Exception as e:
                log(f"  {name} {lname} fold {fi + 1} FAILED: {e}")
        log(f"  {name} fold {fi + 1} done")
    valid = [p for p in [p_rf, p_lgbm, p_xgb] if not np.isnan(p).any()]
    if not valid:
        return None
    return np.mean(valid, axis=0)


def main():
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    log("=== GRAPH MODELS v10 INDUCTIVE (fold-honest) START ===")

    merged, feat_cols, cpt_arr, Fseq, seq_all, _ = assemble_training_frame()
    gold = pd.read_parquet(GOLD)[["LogID","ReadmittedWithin30Days_gold"]]
    gold["LogID"] = gold["LogID"].astype(str); merged["LogID"] = merged["LogID"].astype(str)
    merged = merged.merge(gold, on="LogID", how="left")
    excl = merged["Discharge Disposition"].astype(str).isin(EXCLUDE_LABELS)
    merged = merged.loc[~excl].reset_index(drop=True)
    cpt_arr = np.asarray(cpt_arr).ravel()[~excl.values].reshape(-1, 1)
    y = merged["ReadmittedWithin30Days_gold"].astype(int).values
    N = len(y)
    log(f"cohort N={N}  base rate {y.mean()*100:.2f}%")

    log("assembling tabular blocks (B, C, dispo) — no test-fold leak")
    XB, _ = make_B(merged)
    XC, _ = make_C(merged)
    X_dispo = merged["Discharge Disposition"].astype(str).isin(HOME_LABELS)\
                 .astype(np.float32).values.reshape(-1, 1)

    log("loading raw graph substrate")
    orders_df = collapse_order_runs(load_order_sequence())
    raw_full = load_raw()
    kept = set(_norm_id(merged["LogID"]).values)
    ep = raw_full.enc_prov_edges.copy(); ep["LogID"] = _norm_id(ep["LogID"])
    ep = ep[ep["LogID"].isin(kept)]
    eu = raw_full.enc_unit_edges.copy(); eu["LogID"] = _norm_id(eu["LogID"])
    eu = eu[eu["LogID"].isin(kept)]
    raw = shallow_copy(raw_full)
    raw.enc_prov_edges = ep
    raw.enc_unit_edges = eu

    prov_ids, X_prov, _ = build_provider_features(raw.prov_attrs)
    unit_ids, X_unit, _ = build_unit_features(raw.unit_attrs)
    log(f"  providers {len(prov_ids)}  units {len(unit_ids)}")

    results = []
    oofs = {}

    # =========================================================
    # Row 1: MedHG-PS (original, A3 care-unit graph) — inductive
    # =========================================================
    log("Row 1: MedHG-PS (original, A3 care-unit graph) — INDUCTIVE")
    def _get_medhgps_orig(tr):
        return train_gnn_and_encode_inductive(
            merged, feat_cols, cpt_arr, y, tr,
            raw, prov_ids, X_prov, unit_ids, X_unit,
            fold_seed=SEED, gnn_cfg=GNN_CFG)
    p_orig = run_ensemble_fold_local(merged, feat_cols, cpt_arr, XB, XC,
                                       X_dispo, y, "medhg_orig_inductive",
                                       _get_medhgps_orig)
    if p_orig is not None:
        r = _eval(y, p_orig, "MedHG-PS (original, A3 care-unit) — INDUCTIVE")
        log(f"  {r['model']:50s} AUROC {r['auroc']:.3f} "
            f"({r['auroc_lo']:.3f}-{r['auroc_hi']:.3f}) AUPRC {r['auprc']:.3f}")
        results.append(r); oofs["medhg_orig_inductive"] = p_orig

    # =========================================================
    # Row 2: MedHG-PS (orders as A3) — inductive
    # =========================================================
    log("Row 2: MedHG-PS (orders as A3) — INDUCTIVE")
    raw_og, og_ids, X_og = build_order_group_substrate(raw, merged, orders_df)
    def _get_medhgps_orders(tr):
        return train_gnn_and_encode_inductive(
            merged, feat_cols, cpt_arr, y, tr,
            raw_og, prov_ids, X_prov, og_ids, X_og,
            fold_seed=SEED, gnn_cfg=GNN_CFG)
    p_ord = run_ensemble_fold_local(merged, feat_cols, cpt_arr, XB, XC,
                                     X_dispo, y, "medhg_orders_inductive",
                                     _get_medhgps_orders)
    if p_ord is not None:
        r = _eval(y, p_ord, "MedHG-PS (orders as A3) — INDUCTIVE")
        log(f"  {r['model']:50s} AUROC {r['auroc']:.3f} "
            f"({r['auroc_lo']:.3f}-{r['auroc_hi']:.3f}) AUPRC {r['auprc']:.3f}")
        results.append(r); oofs["medhg_orders_inductive"] = p_ord

    # =========================================================
    # Row 3: Node2Vec + trees — inductive
    # =========================================================
    log("Row 3: Node2Vec + trees — INDUCTIVE")
    gs = precompute_graph(merged, orders_df, raw.enc_prov_edges, raw.prov_attrs)
    def _get_n2v(tr):
        return compute_node2vec_inductive(
            gs, train_ids=tr, dim=64, walks=5, walk_len=10, window=5, seed=SEED)
    p_n2v = run_ensemble_fold_local(merged, feat_cols, cpt_arr, XB, XC,
                                     X_dispo, y, "n2v_inductive", _get_n2v)
    if p_n2v is not None:
        r = _eval(y, p_n2v, "Node2Vec + trees — INDUCTIVE")
        log(f"  {r['model']:50s} AUROC {r['auroc']:.3f} "
            f"({r['auroc_lo']:.3f}-{r['auroc_hi']:.3f}) AUPRC {r['auprc']:.3f}")
        results.append(r); oofs["n2v_inductive"] = p_n2v

    pd.DataFrame(results).to_csv(OUT_RES, index=False)
    np.savez(OUT_OOF, y=y, **oofs)
    log(f"saved {OUT_RES}, {OUT_OOF}")
    log("=== DONE ===")


if __name__ == "__main__":
    main()
