"""Final push v7 — Node2Vec graph embedding + XGBoost / LightGBM.

Closer to G-GBM in spirit than v6:
  v6: extract hand-crafted graph features (centrality, neighborhood risk,
      SVD embedding), feed to standard trees.
  v7: learn Node2Vec walk-based embeddings on the tripartite graph
      (encounter, provider, order_group) and feed to REAL gradient-boosted
      trees (XGBoost and LightGBM) alongside the v6 hand-crafted graph
      features.

The two-stage design keeps splits axis-parallel (like G-GBM/GGBoost is
NOT — those propose true graph-aware split rules), but it does pair a
principled graph-structure encoder (Node2Vec random walks preserving
neighborhood proximity) with real GBDT learners.

Feature blocks:
  A base tabular + CPT (~200)
  B LACE utility (25)
  C LACE + HOSPITAL scores (16)
  H geocode + ACS (6)
  E_graph v6 hand-crafted graph features (18)
  E_n2v Node2Vec encounter embeddings (64)  <- NEW

Learners:
  rf_base   canonical RF (500 trees)
  rf_big    pushed RF (1000 trees, balanced_subsample)
  hgb       HistGradientBoosting (sklearn)
  xgb       XGBoost 3.3
  lgbm      LightGBM 4.6
Ensemble: mean of the top learners' calibrated OOF.
Ablation: drop-one-block on rf_big (v6 showed rf_big is our best single).
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
import networkx as nx
from node2vec import Node2Vec
from scipy.sparse import csr_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, f1_score)
import xgboost as xgb
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.final_push_080 import (
    log, make_A, make_B, make_C, make_H,
    learner_rf, learner_hgb, eval_pooled_oof,
    SEED, DATA_DIR, GOLD,
)
from analysis.final_push_080_v6 import precompute_graph, build_Egraph

import medhg_ps.config as C, medhg_ps.data as d
from medhg_ps.data import load_raw
from medhg_ps.deploy import assemble_training_frame

OUT_LOG = Path("artifacts/newdata/final_push_080_v7.log")
OUT_RES = Path("artifacts/newdata/final_push_080_v7_results.csv")
OUT_OOF = Path("artifacts/newdata/final_push_080_v7_oof.npz")
OUT_ABL = Path("artifacts/newdata/final_push_080_v7_ablation.csv")

N2V_DIM = 64
N2V_WALK_LEN = 10
N2V_NUM_WALKS = 5
N2V_WINDOW = 5


# ============================================================
# Node2Vec on the tripartite graph -- built ONCE (structural signal
# has no target leakage since walks don't use y).
# ============================================================
def _build_adjacency(gs):
    """Build a fast CSR-based adjacency for our tripartite graph.
    Nodes: encounter[0..N_enc), provider[N_enc..N_enc+N_prov),
           order_group[N_enc+N_prov..).
    Returns (indptr, indices, num_nodes, N_enc)."""
    N_enc = gs["N"]
    N_prov = len(gs["prov_tokens"])
    N_og = len(gs["og_tokens"])
    total = N_enc + N_prov + N_og

    # collect edges
    a2_prov = gs["a2_prov_per_enc"]
    prov2id = gs["prov2id"]
    rows, cols = [], []
    for enc_i, provs in a2_prov.items():
        for p in provs:
            if p in prov2id:
                pj = N_enc + prov2id[p]
                rows.append(enc_i); cols.append(pj)
                rows.append(pj); cols.append(enc_i)

    M_eo_coo = gs["M_eo"].tocoo()
    for i, k in zip(M_eo_coo.row, M_eo_coo.col):
        og_j = N_enc + N_prov + int(k)
        rows.append(int(i)); cols.append(og_j)
        rows.append(og_j); cols.append(int(i))

    idx_sort = np.argsort(np.asarray(rows))
    rows_s = np.asarray(rows)[idx_sort]
    cols_s = np.asarray(cols)[idx_sort]
    indptr = np.zeros(total + 1, dtype=np.int64)
    np.add.at(indptr, rows_s + 1, 1)
    np.cumsum(indptr, out=indptr)
    indices = cols_s.astype(np.int64)
    return indptr, indices, total, N_enc


def _random_walks(indptr, indices, N_enc, num_nodes,
                  walks_per_node, walk_len, seed):
    """Fast unbiased random walks (p=q=1). Returns list of walks
    where each element is a list of node ids as strings."""
    rng = np.random.default_rng(seed)
    walks = []
    starts = np.arange(num_nodes, dtype=np.int64)
    for _ in range(walks_per_node):
        rng.shuffle(starts)
        for s in starts:
            walk = [s]
            cur = s
            for _step in range(walk_len - 1):
                lo, hi = indptr[cur], indptr[cur + 1]
                if hi == lo:
                    break
                cur = int(indices[rng.integers(lo, hi)])
                walk.append(cur)
            walks.append([str(w) for w in walk])
    return walks


def compute_n2v_embeddings(gs, workers=4, seed=SEED):
    """Fast Node2Vec-style embedding via NumPy walks + gensim Word2Vec.
    Skips the slow node2vec library entirely.
    Returns (N, N2V_DIM) matrix aligned to gs['all_logids']."""
    log("  building fast CSR adjacency")
    indptr, indices, total, N_enc = _build_adjacency(gs)
    log(f"  graph: nodes {total:,}  encounters {N_enc:,}  edges {len(indices)//2:,}")

    log(f"  generating walks (walks={N2V_NUM_WALKS}, walk_len={N2V_WALK_LEN})")
    t0 = time.time()
    walks = _random_walks(indptr, indices, N_enc, total,
                          walks_per_node=N2V_NUM_WALKS,
                          walk_len=N2V_WALK_LEN, seed=seed)
    log(f"  {len(walks):,} walks in {time.time()-t0:.1f}s")

    log(f"  fitting Word2Vec skip-gram (dim={N2V_DIM}, window={N2V_WINDOW})")
    from gensim.models import Word2Vec
    t0 = time.time()
    w2v = Word2Vec(
        sentences=walks, vector_size=N2V_DIM, window=N2V_WINDOW,
        min_count=1, sg=1, workers=workers, seed=seed,
        epochs=5,
    )
    log(f"  Word2Vec fit in {time.time()-t0:.1f}s  vocab {len(w2v.wv.key_to_index):,}")

    emb = np.zeros((N_enc, N2V_DIM), dtype=np.float32)
    for i in range(N_enc):
        key = str(i)
        if key in w2v.wv:
            emb[i] = w2v.wv[key]
    log(f"  encounter embeddings {emb.shape}  norm-mean {np.linalg.norm(emb, axis=1).mean():.3f}")
    return emb


# ============================================================
# XGBoost / LightGBM wrappers (calibratable)
# ============================================================
def learner_xgb(seed=SEED):
    return xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.05,
        max_depth=6, min_child_weight=5,
        subsample=0.85, colsample_bytree=0.75,
        reg_lambda=1.0, tree_method="hist",
        eval_metric="logloss", random_state=seed,
        n_jobs=-1, use_label_encoder=False, verbosity=0,
    )


def learner_lgbm(seed=SEED):
    return lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.05,
        num_leaves=63, min_child_samples=15,
        subsample=0.85, colsample_bytree=0.75,
        reg_lambda=1.0, class_weight="balanced",
        random_state=seed, n_jobs=-1, verbosity=-1,
    )


def main():
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    log("=== FINAL PUSH v7 (Node2Vec + XGBoost/LightGBM) START ===")

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

    log("precomputing graph structure")
    from medhg_ps.data import load_order_sequence, collapse_order_runs
    orders_df = collapse_order_runs(load_order_sequence())
    raw = load_raw()
    gs = precompute_graph(merged, orders_df, raw.enc_prov_edges, raw.prov_attrs)

    # Node2Vec embeddings — built once, no target leakage (walks are label-independent)
    log("BLOCK E_n2v: computing Node2Vec encounter embeddings (one-time)")
    t0 = time.time()
    X_n2v = compute_n2v_embeddings(gs)
    log(f"  E_n2v built in {time.time()-t0:.1f}s")

    log("=== CROSS VALIDATION (5-fold seed 42) ===")
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    p_oof = {"rf_base": np.full(N, np.nan),
             "rf_big":  np.full(N, np.nan),
             "hgb":     np.full(N, np.nan),
             "xgb":     np.full(N, np.nan),
             "lgbm":    np.full(N, np.nan)}
    p_ab_no_Egraph = np.full(N, np.nan)
    p_ab_no_En2v   = np.full(N, np.nan)
    p_ab_no_graphs = np.full(N, np.nan)

    for fi, (tr, te) in enumerate(skf.split(np.zeros(N), y)):
        log(f"--- fold {fi + 1}/5  train {len(tr)} test {len(te)} ---")
        train_mask = np.zeros(N, bool); train_mask[tr] = True

        XA = make_A(merged, feat_cols, cpt_arr, train_mask)
        log(f"  A {XA.shape}")

        log("  building E_graph features (leak-safe)")
        XEg = build_Egraph(gs, y, train_mask)

        X_full = np.hstack([XA, XB, XC, XH, XEg, X_n2v]).astype(np.float32)
        X_no_Eg  = np.hstack([XA, XB, XC, XH, X_n2v]).astype(np.float32)
        X_no_n2v = np.hstack([XA, XB, XC, XH, XEg]).astype(np.float32)
        X_no_all = np.hstack([XA, XB, XC, XH]).astype(np.float32)
        if fi == 0:
            log(f"  X_full {X_full.shape}  X_no_Egraph {X_no_Eg.shape}  "
                f"X_no_n2v {X_no_n2v.shape}  X_no_graphs {X_no_all.shape}")

        for name, mk in [("rf_base", lambda: learner_rf()),
                         ("rf_big",  lambda: learner_rf(big=True)),
                         ("hgb",     lambda: learner_hgb()),
                         ("xgb",     lambda: learner_xgb()),
                         ("lgbm",    lambda: learner_lgbm())]:
            try:
                est = CalibratedClassifierCV(
                    mk(), method="isotonic", cv=3).fit(X_full[tr], y[tr])
                p_oof[name][te] = est.predict_proba(X_full[te])[:, 1]
                log(f"  fold {fi + 1} {name} done")
            except Exception as e:
                log(f"  fold {fi + 1} {name} FAILED: {e}")

        # Ablation on rf_big (v6's best single learner)
        for Xab, arr, tag in [(X_no_Eg,  p_ab_no_Egraph, "no_Egraph"),
                              (X_no_n2v, p_ab_no_En2v,   "no_En2v"),
                              (X_no_all, p_ab_no_graphs, "no_graphs")]:
            try:
                est = CalibratedClassifierCV(
                    learner_rf(big=True), method="isotonic",
                    cv=3).fit(Xab[tr], y[tr])
                arr[te] = est.predict_proba(Xab[te])[:, 1]
                log(f"  ablation {tag} done")
            except Exception as e:
                log(f"  ablation {tag} FAILED: {e}")

    log("=== POOLED OOF EVAL ===")
    results = []
    for name in ["rf_base", "rf_big", "hgb", "xgb", "lgbm"]:
        p = p_oof[name]
        if np.isnan(p).any():
            log(f"  {name} has NaN, skip"); continue
        r = eval_pooled_oof(y, p, name)
        log(f"  {name:10s} AUROC {r['auroc']:.3f} "
            f"({r['auroc_lo']:.3f}-{r['auroc_hi']:.3f}) "
            f"AUPRC {r['auprc']:.3f} F1 {r['f1']:.3f}")
        results.append(r)

    valid = [k for k in p_oof if not np.isnan(p_oof[k]).any()]
    if len(valid) >= 2:
        p_ens = np.mean([p_oof[k] for k in valid], axis=0)
        r = eval_pooled_oof(y, p_ens, "ensemble_v7_all")
        log(f"  ensemble_v7_all   AUROC {r['auroc']:.3f} "
            f"({r['auroc_lo']:.3f}-{r['auroc_hi']:.3f}) "
            f"AUPRC {r['auprc']:.3f} F1 {r['f1']:.3f}")
        results.append(r)

        # top-3 by AUROC
        top3 = sorted([(name, roc_auc_score(y, p_oof[name])) for name in valid],
                      key=lambda t: -t[1])[:3]
        top3_names = [t[0] for t in top3]
        p_ens_top3 = np.mean([p_oof[k] for k in top3_names], axis=0)
        r = eval_pooled_oof(y, p_ens_top3, f"ensemble_v7_top3_{'+'.join(top3_names)}")
        log(f"  ensemble_v7_top3  AUROC {r['auroc']:.3f} "
            f"({r['auroc_lo']:.3f}-{r['auroc_hi']:.3f}) "
            f"AUPRC {r['auprc']:.3f} F1 {r['f1']:.3f}")
        results.append(r)

    log("=== ABLATION (rf_big only) ===")
    ablation = []
    for tag, p in [("rf_big_full",       p_oof["rf_big"]),
                   ("rf_big_no_Egraph",  p_ab_no_Egraph),
                   ("rf_big_no_En2v",    p_ab_no_En2v),
                   ("rf_big_no_graphs",  p_ab_no_graphs)]:
        if np.isnan(p).any():
            log(f"  {tag} has NaN, skip"); continue
        au = roc_auc_score(y, p); ap = average_precision_score(y, p)
        ablation.append(dict(config=tag, auroc=au, auprc=ap))
        log(f"  {tag:22s} AUROC {au:.3f}  AUPRC {ap:.3f}")

    pd.DataFrame(results).to_csv(OUT_RES, index=False)
    pd.DataFrame(ablation).to_csv(OUT_ABL, index=False)
    np.savez(OUT_OOF, y=y,
             p_ab_no_Egraph=p_ab_no_Egraph,
             p_ab_no_En2v=p_ab_no_En2v,
             p_ab_no_graphs=p_ab_no_graphs,
             **p_oof)
    log(f"saved {OUT_RES}, {OUT_ABL}, {OUT_OOF}")
    log("=== DONE ===")


if __name__ == "__main__":
    main()
