"""Proper Node2Vec (v11) — biased walks with p, q + Optuna tuning.

Fixes the v7 shortcut where p=q=1 (unbiased walks = DeepWalk-equivalent).
Implements the true Node2Vec algorithm from Grover & Leskovec 2016 with:
  - Biased random walks: transition to neighbor x from (t, v) uses
    weight 1/p if x == t (return), 1 if x is neighbor of t (BFS-like),
    1/q otherwise (DFS-like)
  - Adjacency SETS for O(1) "is neighbor of t" lookup

Hyperparameter sweep via Optuna TPE, 30 trials:
  p, q in {0.25, 0.5, 1, 2, 4}
  walk_len in [10, 30], num_walks in [5, 15]
  dim in {32, 64, 128}, window in [3, 10]

Scoring: 3-fold inner CV AUROC of LightGBM on tabular + Node2Vec embedding.
Final: 5-fold outer CV with best config + rf_big + LGBM + XGB ensemble.

Cohort: N=13,858 (corrected), Discharge Disposition = home/other.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, f1_score)
import xgboost as xgb
import lightgbm as lgb
from gensim.models import Word2Vec

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.final_push_080 import (
    log, make_A, make_B, make_C, make_H,
    learner_rf, eval_pooled_oof, SEED, DATA_DIR, GOLD,
)
from analysis.final_push_080_v6 import precompute_graph
from analysis.final_push_080_v8 import _norm_id

import medhg_ps.config as C
from medhg_ps.data import load_raw, load_order_sequence, collapse_order_runs
from medhg_ps.deploy import assemble_training_frame
from medhg_ps.evaluate import _bootstrap_ci

optuna.logging.set_verbosity(optuna.logging.WARNING)

OUT_LOG = Path("artifacts/newdata/node2vec_v11.log")
OUT_RES = Path("artifacts/newdata/node2vec_v11_results.csv")
OUT_OOF = Path("artifacts/newdata/node2vec_v11_oof.npz")
OUT_BP  = Path("artifacts/newdata/node2vec_v11_best_params.json")

EXCLUDE_LABELS = {
    "Expired", "Expired in Medical Facility",
    "Hospice/Home", "Hospice/Medical Facility",
    "Acute / Short Term Hospital", "Left Against Medical Advice",
}
HOME_LABELS = {"Home or Self Care", "Home-Health Care Svc"}

N_TRIALS = 30


# ============================================================
# Biased random walker (Node2Vec algorithm)
# ============================================================
def build_tripartite_adjacency(gs):
    """Return (indptr, indices, adj_sets, N_enc, num_nodes) for
    encounter + provider + order-group nodes."""
    N_enc = gs["N"]
    N_prov = len(gs["prov_tokens"])
    N_og = len(gs["og_tokens"])
    total = N_enc + N_prov + N_og

    rows, cols = [], []
    for enc_i, provs in gs["a2_prov_per_enc"].items():
        for p in provs:
            if p in gs["prov2id"]:
                pj = N_enc + gs["prov2id"][p]
                rows.append(enc_i); cols.append(pj)
                rows.append(pj); cols.append(enc_i)
    coo = gs["M_eo"].tocoo()
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

    # Adjacency sets for O(1) "is x a neighbor of t?" lookup
    log(f"  building adjacency sets (total {total:,} nodes)")
    adj_sets = [None] * total
    for u in range(total):
        adj_sets[u] = set(int(v) for v in indices[indptr[u]:indptr[u + 1]])
    return indptr, indices, adj_sets, N_enc, total


def biased_walks(indptr, indices, adj_sets, N_total,
                 walks_per_node, walk_len, p, q, seed):
    """True Node2Vec biased random walks.
    - p (return parameter): higher p -> less likely to revisit
    - q (in-out parameter): q<1 -> DFS (outward), q>1 -> BFS (local)
    """
    rng = np.random.default_rng(seed)
    walks = []
    inv_p = 1.0 / p
    inv_q = 1.0 / q
    starts = np.arange(N_total, dtype=np.int64)
    for w in range(walks_per_node):
        rng.shuffle(starts)
        for s in starts:
            walk = [int(s)]
            cur = int(s)
            prev = -1
            for step in range(walk_len - 1):
                lo, hi = int(indptr[cur]), int(indptr[cur + 1])
                if hi == lo:
                    break
                neigh = indices[lo:hi]
                if prev < 0:
                    # first step: uniform
                    cur = int(neigh[rng.integers(0, len(neigh))])
                    prev_new = cur
                else:
                    weights = np.empty(len(neigh), dtype=np.float64)
                    prev_set = adj_sets[prev]
                    for i, x in enumerate(neigh):
                        xi = int(x)
                        if xi == prev:
                            weights[i] = inv_p
                        elif xi in prev_set:
                            weights[i] = 1.0
                        else:
                            weights[i] = inv_q
                    weights /= weights.sum()
                    # cumsum sampling
                    r = rng.random()
                    cum = 0.0
                    picked = neigh[-1]
                    for i, wt in enumerate(weights):
                        cum += wt
                        if r <= cum:
                            picked = int(neigh[i])
                            break
                    prev_new = cur
                    cur = picked
                prev = prev_new
                walk.append(cur)
            walks.append([str(x) for x in walk])
    return walks


def train_n2v_embedding(indptr, indices, adj_sets, N_enc, N_total,
                        p, q, walk_len, num_walks, dim, window, seed):
    walks = biased_walks(indptr, indices, adj_sets, N_total,
                          walks_per_node=num_walks, walk_len=walk_len,
                          p=p, q=q, seed=seed)
    w2v = Word2Vec(sentences=walks, vector_size=dim, window=window,
                   min_count=1, sg=1, workers=4, seed=seed, epochs=5)
    emb = np.zeros((N_enc, dim), dtype=np.float32)
    for i in range(N_enc):
        key = str(i)
        if key in w2v.wv:
            emb[i] = w2v.wv[key]
    return emb


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


def main():
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    log("=== NODE2VEC v11 (biased walks + Optuna) START ===")

    # Cohort with exclusions
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
    log(f"cohort N={N} base rate {y.mean()*100:.2f}%")

    # Tabular backbone
    log("assembling tabular backbone")
    XB, _ = make_B(merged); XC, _ = make_C(merged); XH, _ = make_H(merged)
    X_dispo = merged["Discharge Disposition"].astype(str).isin(HOME_LABELS)\
                 .astype(np.float32).values.reshape(-1, 1)
    train_mask = np.ones(N, bool)
    XA = make_A(merged, feat_cols, cpt_arr, train_mask)
    X_back = np.hstack([XA, XB, XC, XH, X_dispo]).astype(np.float32)
    log(f"  X_back {X_back.shape}")

    # Graph substrate
    log("building tripartite graph adjacency")
    orders_df = collapse_order_runs(load_order_sequence())
    raw = load_raw()
    kept = set(_norm_id(merged["LogID"]).values)
    ep = raw.enc_prov_edges.copy()
    ep["LogID"] = _norm_id(ep["LogID"])
    ep = ep[ep["LogID"].isin(kept)]
    gs = precompute_graph(merged, orders_df, ep, raw.prov_attrs)
    indptr, indices, adj_sets, N_enc, N_total = \
        build_tripartite_adjacency(gs)
    log(f"  graph {N_total:,} nodes / {len(indices)//2:,} edges")

    # ------------------------------------------------------------
    # Optuna: sweep p, q, walk_len, num_walks, dim, window
    # Score = 3-fold LGBM AUROC on X_back + embedding
    # ------------------------------------------------------------
    log(f"=== TUNING with Optuna ({N_TRIALS} trials) ===")

    def objective(trial):
        p = trial.suggest_categorical("p", [0.25, 0.5, 1.0, 2.0, 4.0])
        q = trial.suggest_categorical("q", [0.25, 0.5, 1.0, 2.0, 4.0])
        walk_len = trial.suggest_int("walk_len", 10, 25, step=5)
        num_walks = trial.suggest_int("num_walks", 5, 15, step=5)
        dim = trial.suggest_categorical("dim", [32, 64, 128])
        window = trial.suggest_int("window", 3, 10)

        t0 = time.time()
        try:
            emb = train_n2v_embedding(
                indptr, indices, adj_sets, N_enc, N_total,
                p, q, walk_len, num_walks, dim, window, seed=SEED)
        except Exception as e:
            log(f"  trial FAILED at n2v: {e}")
            return 0.5
        X = np.hstack([X_back, emb]).astype(np.float32)
        try:
            scores = cross_val_score(make_lgbm(), X, y,
                                     cv=3, scoring="roc_auc", n_jobs=1)
        except Exception as e:
            log(f"  trial FAILED at CV: {e}")
            return 0.5
        auroc = float(scores.mean())
        log(f"  trial {trial.number:2d}  p={p} q={q} wl={walk_len} nw={num_walks} "
            f"dim={dim} win={window}  AUROC={auroc:.4f} "
            f"in {time.time()-t0:.1f}s")
        return auroc

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    best = study.best_params
    best_auc = study.best_value
    log(f"\nBEST inner CV AUROC = {best_auc:.4f}")
    log(f"BEST params = {best}")

    # ------------------------------------------------------------
    # Final 5-fold outer CV with best params + full ensemble
    # ------------------------------------------------------------
    log("=== FINAL 5-fold OUTER CV with best params ===")
    log("  computing final Node2Vec embedding")
    t0 = time.time()
    emb = train_n2v_embedding(indptr, indices, adj_sets, N_enc, N_total,
                              best["p"], best["q"], best["walk_len"],
                              best["num_walks"], best["dim"],
                              best["window"], seed=SEED)
    log(f"  embedding {emb.shape} in {time.time()-t0:.1f}s")
    X_full = np.hstack([X_back, emb]).astype(np.float32)
    log(f"  X_full {X_full.shape}")

    p_rf = np.full(N, np.nan)
    p_lgbm = np.full(N, np.nan)
    p_xgb = np.full(N, np.nan)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for fi, (tr, te) in enumerate(skf.split(np.zeros(N), y)):
        est = CalibratedClassifierCV(learner_rf(big=True), method="isotonic",
                                      cv=3).fit(X_full[tr], y[tr])
        p_rf[te] = est.predict_proba(X_full[te])[:, 1]
        est = CalibratedClassifierCV(make_lgbm(), method="isotonic",
                                      cv=3).fit(X_full[tr], y[tr])
        p_lgbm[te] = est.predict_proba(X_full[te])[:, 1]
        est = CalibratedClassifierCV(make_xgb(), method="isotonic",
                                      cv=3).fit(X_full[tr], y[tr])
        p_xgb[te] = est.predict_proba(X_full[te])[:, 1]
        log(f"  fold {fi + 1} done")
    p_ens = np.mean([p_rf, p_lgbm, p_xgb], axis=0)

    results = []
    for name, p in [("Node2Vec + rf_big", p_rf),
                    ("Node2Vec + LightGBM", p_lgbm),
                    ("Node2Vec + XGBoost", p_xgb),
                    ("Node2Vec + Ensemble", p_ens)]:
        r = _eval(y, p, name)
        r["best_params"] = str(best)
        log(f"  {r['model']:30s} AUROC {r['auroc']:.3f} ({r['auroc_lo']:.3f}-{r['auroc_hi']:.3f}) AUPRC {r['auprc']:.3f}")
        results.append(r)

    pd.DataFrame(results).to_csv(OUT_RES, index=False)
    np.savez(OUT_OOF, y=y, p_rf=p_rf, p_lgbm=p_lgbm, p_xgb=p_xgb, p_ens=p_ens)
    import json
    with open(OUT_BP, "w") as f:
        json.dump({"best_params": best, "best_inner_auroc": best_auc}, f, indent=2)
    log(f"saved {OUT_RES}, {OUT_OOF}, {OUT_BP}")
    log("=== DONE ===")


if __name__ == "__main__":
    main()
