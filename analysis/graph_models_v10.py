"""Graph-model rerun on cohort-corrected N=13,858 with dispo.

Runs four graph configurations, each fed to the same downstream tree
ensemble (rf_big + LightGBM + XGBoost, mean of three calibrated OOFs).
Tabular backbone: A + B + C + H + dispo (same as tabular_leaders_v10).

Rows:
  1. MedHG-PS (original)      — ie-HGCN on encounter+provider+CARE-UNIT graph
  2. MedHG-PS (orders as A3)  — ie-HGCN on encounter+provider+ORDER-GROUP graph
  3. Node2Vec + trees         — Node2Vec walks on encounter+prov+order-group
  4. Tree-based graph (v8)    — v8 rich node design: 5 provider roles, 5 order
                                 categories, 17 Charlson diagnoses, top-200
                                 order-set bundles; features = degree + label
                                 propagation + SVD-16 + Node2Vec-64

Same 5-fold seed 42, isotonic-calibrated pooled OOF, bootstrap n=2000 CIs.
"""
from __future__ import annotations
import sys, time, warnings
from copy import copy
from dataclasses import replace
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, f1_score, precision_score,
                             recall_score)
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.final_push_080 import (
    log, make_A, make_B, make_C, make_H,
    learner_rf, eval_pooled_oof, SEED, DATA_DIR, GOLD,
)
from analysis.final_push_080_v6 import build_Egraph
from analysis.final_push_080_v7 import (_build_adjacency, _random_walks)
from analysis.final_push_080_v8 import (
    build_rich_graph_substrate, build_Egraph_v8, compute_n2v_v8,
    _norm_id, prov_role, PROV_ROLES,
)
from analysis.final_push_080_v2 import (train_gnn_and_encode, GNN_CFG,
                                          GNN_DEV, VAL_FRAC)
from analysis.final_push_080_v3 import build_order_group_substrate

import medhg_ps.config as C, medhg_ps.data as d
from medhg_ps.data import (load_raw, load_order_sequence, collapse_order_runs,
                            build_provider_features, build_unit_features)
from medhg_ps.deploy import assemble_training_frame
from medhg_ps.evaluate import _bootstrap_ci
from gensim.models import Word2Vec

OUT_LOG = Path("artifacts/newdata/graph_models_v10.log")
OUT_RES = Path("artifacts/newdata/graph_models_v10_results.csv")
OUT_OOF = Path("artifacts/newdata/graph_models_v10_oof.npz")

EXCLUDE_LABELS = {
    "Expired", "Expired in Medical Facility",
    "Hospice/Home", "Hospice/Medical Facility",
    "Acute / Short Term Hospital", "Left Against Medical Advice",
}
HOME_LABELS = {"Home or Self Care", "Home-Health Care Svc"}


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


def run_ensemble_cv(X_full, y, name, get_extra_features=None):
    """Run 5-fold with rf_big + LGBM + XGB, ensemble = mean.
    get_extra_features(train_idx) returns a per-fold N x D feature block
    (or None). If given, extra features are concatenated to X_full each fold."""
    N = len(y)
    p_rf = np.full(N, np.nan)
    p_lgbm = np.full(N, np.nan)
    p_xgb = np.full(N, np.nan)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for fi, (tr, te) in enumerate(skf.split(np.zeros(N), y)):
        if get_extra_features is not None:
            X_extra = get_extra_features(tr)
            X = np.hstack([X_full, X_extra]).astype(np.float32)
        else:
            X = X_full.astype(np.float32)
        try:
            est = CalibratedClassifierCV(learner_rf(big=True),
                                          method="isotonic", cv=3).fit(X[tr], y[tr])
            p_rf[te] = est.predict_proba(X[te])[:, 1]
        except Exception as e:
            log(f"  {name} rf fold {fi + 1} FAILED: {e}")
        try:
            est = CalibratedClassifierCV(make_lgbm(),
                                          method="isotonic", cv=3).fit(X[tr], y[tr])
            p_lgbm[te] = est.predict_proba(X[te])[:, 1]
        except Exception as e:
            log(f"  {name} lgbm fold {fi + 1} FAILED: {e}")
        try:
            est = CalibratedClassifierCV(make_xgb(),
                                          method="isotonic", cv=3).fit(X[tr], y[tr])
            p_xgb[te] = est.predict_proba(X[te])[:, 1]
        except Exception as e:
            log(f"  {name} xgb fold {fi + 1} FAILED: {e}")
        log(f"  {name} fold {fi + 1} done")

    valid = [p for p in [p_rf, p_lgbm, p_xgb] if not np.isnan(p).any()]
    if not valid:
        return None
    return np.mean(valid, axis=0)


def main():
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    log("=== GRAPH MODELS v10 (cohort-corrected + dispo) START ===")

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

    # ---- Tabular backbone ----
    log("assembling tabular backbone (A + B + C + H + dispo)")
    XB, _ = make_B(merged)
    XC, _ = make_C(merged)
    XH, _ = make_H(merged)
    train_mask = np.ones(N, bool)
    XA = make_A(merged, feat_cols, cpt_arr, train_mask)
    X_dispo = merged["Discharge Disposition"].astype(str).isin(HOME_LABELS)\
                 .astype(np.float32).values.reshape(-1, 1)
    X_backbone = np.hstack([XA, XB, XC, XH, X_dispo]).astype(np.float32)
    log(f"  X_backbone {X_backbone.shape}")

    # ---- Load raw graph tables + orders ----
    log("loading raw graph substrate")
    orders_df = collapse_order_runs(load_order_sequence())
    raw_full = load_raw()

    # Filter raw tables to kept encounters
    kept_logids = set(_norm_id(merged["LogID"]).values)
    ep = raw_full.enc_prov_edges.copy()
    ep["LogID"] = _norm_id(ep["LogID"])
    ep = ep[ep["LogID"].isin(kept_logids)]
    eu = raw_full.enc_unit_edges.copy()
    eu["LogID"] = _norm_id(eu["LogID"])
    eu = eu[eu["LogID"].isin(kept_logids)]
    log(f"  ep {len(ep):,}  eu {len(eu):,}")
    raw = copy(raw_full)
    raw.enc_prov_edges = ep
    raw.enc_unit_edges = eu

    results = []
    oofs = {}

    # -----------------------------------------------------------
    # Row 1: MedHG-PS (original) — care-unit A3
    # -----------------------------------------------------------
    log("Row 1: MedHG-PS (original) — encounter+provider+unit graph")
    prov_ids, X_prov, _ = build_provider_features(raw.prov_attrs)
    unit_ids, X_unit, _ = build_unit_features(raw.unit_attrs)
    log(f"  providers {len(prov_ids)}  units {len(unit_ids)}")

    def _get_medhgps_orig(tr):
        return train_gnn_and_encode(
            merged, feat_cols, cpt_arr, y, tr, raw, prov_ids, X_prov,
            unit_ids, X_unit, fold_seed=SEED)
    p_medhg_orig = run_ensemble_cv(X_backbone, y, "medhg_orig",
                                    _get_medhgps_orig)
    if p_medhg_orig is not None:
        r = _eval(y, p_medhg_orig, "MedHG-PS (original)")
        log(f"  {r['model']:35s} AUROC {r['auroc']:.3f} ({r['auroc_lo']:.3f}-{r['auroc_hi']:.3f}) AUPRC {r['auprc']:.3f}")
        results.append(r); oofs["medhg_orig"] = p_medhg_orig

    # -----------------------------------------------------------
    # Row 2: MedHG-PS (orders as A3)
    # -----------------------------------------------------------
    log("Row 2: MedHG-PS (orders as A3) — encounter+provider+ordergroup")
    raw_og, og_ids, X_og = build_order_group_substrate(raw, merged, orders_df)

    def _get_medhgps_orders(tr):
        return train_gnn_and_encode(
            merged, feat_cols, cpt_arr, y, tr, raw_og, prov_ids, X_prov,
            og_ids, X_og, fold_seed=SEED)
    p_medhg_ord = run_ensemble_cv(X_backbone, y, "medhg_orders",
                                   _get_medhgps_orders)
    if p_medhg_ord is not None:
        r = _eval(y, p_medhg_ord, "MedHG-PS (orders as A3)")
        log(f"  {r['model']:35s} AUROC {r['auroc']:.3f} ({r['auroc_lo']:.3f}-{r['auroc_hi']:.3f}) AUPRC {r['auprc']:.3f}")
        results.append(r); oofs["medhg_orders"] = p_medhg_ord

    # -----------------------------------------------------------
    # Row 3: Node2Vec + trees (v7 style)
    # -----------------------------------------------------------
    log("Row 3: Node2Vec + trees — tripartite graph")
    # Use v6's precompute_graph for the substrate
    from analysis.final_push_080_v6 import precompute_graph
    gs_simple = precompute_graph(merged, orders_df,
                                  raw.enc_prov_edges, raw.prov_attrs)
    # Build fast walker
    N_enc = gs_simple["N"]
    N_prov = len(gs_simple["prov_tokens"])
    N_og = len(gs_simple["og_tokens"])
    total = N_enc + N_prov + N_og

    # Compute Node2Vec ONCE (transductive)
    rows, cols = [], []
    for enc_i, provs in gs_simple["a2_prov_per_enc"].items():
        for p in provs:
            if p in gs_simple["prov2id"]:
                pj = N_enc + gs_simple["prov2id"][p]
                rows.append(enc_i); cols.append(pj)
                rows.append(pj); cols.append(enc_i)
    coo = gs_simple["M_eo"].tocoo()
    for i, k in zip(coo.row, coo.col):
        og_j = N_enc + N_prov + int(k)
        rows.append(int(i)); cols.append(og_j)
        rows.append(og_j); cols.append(int(i))
    order = np.argsort(np.asarray(rows))
    rows_s = np.asarray(rows)[order]
    cols_s = np.asarray(cols)[order]
    indptr = np.zeros(total + 1, dtype=np.int64)
    np.add.at(indptr, rows_s + 1, 1)
    np.cumsum(indptr, out=indptr)
    indices = cols_s.astype(np.int64)
    log(f"  graph {total:,} nodes, {len(rows)//2:,} edges")
    log(f"  generating walks")
    walks = _random_walks(indptr, indices, N_enc, total,
                          walks_per_node=5, walk_len=10, seed=SEED)
    log(f"  {len(walks):,} walks; fitting Word2Vec")
    w2v = Word2Vec(sentences=walks, vector_size=64, window=5,
                   min_count=1, sg=1, workers=4, seed=SEED, epochs=5)
    X_n2v = np.zeros((N_enc, 64), dtype=np.float32)
    for i in range(N_enc):
        k = str(i)
        if k in w2v.wv:
            X_n2v[i] = w2v.wv[k]
    log(f"  X_n2v {X_n2v.shape}")

    X_n2v_full = np.hstack([X_backbone, X_n2v]).astype(np.float32)
    p_n2v = run_ensemble_cv(X_n2v_full, y, "n2v_trees", None)
    if p_n2v is not None:
        r = _eval(y, p_n2v, "Node2Vec + trees")
        log(f"  {r['model']:35s} AUROC {r['auroc']:.3f} ({r['auroc_lo']:.3f}-{r['auroc_hi']:.3f}) AUPRC {r['auprc']:.3f}")
        results.append(r); oofs["n2v_trees"] = p_n2v

    # -----------------------------------------------------------
    # Row 4: Tree-based graph (v8 rich node design)
    # -----------------------------------------------------------
    log("Row 4: Tree-based graph — v8 rich node design")
    gs_rich = build_rich_graph_substrate(merged, orders_df,
                                          raw.enc_prov_edges, raw.prov_attrs)
    # Node2Vec on multi-type graph
    X_n2v_rich = compute_n2v_v8(gs_rich)
    log(f"  X_n2v_rich {X_n2v_rich.shape}")

    def _get_rich_graph_features(tr):
        train_mask = np.zeros(N, bool); train_mask[tr] = True
        E = build_Egraph_v8(gs_rich, y, train_mask)
        return np.hstack([E["E_deg"], E["E_lp"], E["E_svd"],
                          X_n2v_rich]).astype(np.float32)
    p_rich = run_ensemble_cv(X_backbone, y, "tree_graph_v8",
                              _get_rich_graph_features)
    if p_rich is not None:
        r = _eval(y, p_rich, "Tree-based graph (v8 nodes)")
        log(f"  {r['model']:35s} AUROC {r['auroc']:.3f} ({r['auroc_lo']:.3f}-{r['auroc_hi']:.3f}) AUPRC {r['auprc']:.3f}")
        results.append(r); oofs["tree_graph_v8"] = p_rich

    pd.DataFrame(results).to_csv(OUT_RES, index=False)
    np.savez(OUT_OOF, y=y, **oofs)
    log(f"saved {OUT_RES}, {OUT_OOF}")
    log("=== DONE ===")


if __name__ == "__main__":
    main()
