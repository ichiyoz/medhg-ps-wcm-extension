"""Final push v6 — Tree-based graph algorithm approach.

Instead of GNN message passing, we extract PATH-LEVEL / NEIGHBORHOOD-LEVEL
features from the encounter <-> provider <-> order-group graph and feed
them to gradient-boosted trees. This mirrors the spirit of G-GBM
(Graph Gradient Boosting Machine) and GGBoost, which extend GBDT to
operate on relational data by encoding graph structure as tabular
features.

Feature blocks:
  A base tabular + CPT (~200)
  B LACE utility (25)
  C LACE + HOSPITAL scores (16)
  H geocode + ACS (6)
  E_graph graph-structural features (NEW, ~20)

E_graph (LEAK-SAFE — only train-fold labels are ever used for neighbor
outcome averaging):
  1. Provider-shared neighborhood readmission rate
     - For each encounter e, find TRAIN encounters that share >=1 provider.
       Compute the mean gold label of those TRAIN encounters. This is a
       label-propagation-style "your surgical team's other patients tend
       to bounce back" signal.
     - Repeat at threshold >=2 providers (tighter neighborhood).
  2. Order-group-shared neighborhood readmission rate
     - Same as above but neighbors = share >= K order-groups.
     - Thresholds K = 3, 5, 8.
  3. Provider centrality features
     - Number of providers on encounter
     - Number of DISTINCT providers on encounter
     - Max provider case-volume-2yr (from A4)
  4. Order-group diversity
     - Number of distinct order-groups
     - Shannon entropy of order-group frequencies
     - Fraction of top-3 orders that are "SLP/PT/OT/rehab" (high-risk marker)
  5. Bipartite embedding features (LSA-style, LEAK-SAFE)
     - TruncatedSVD on the encounter x order_group co-occurrence matrix (train
       rows only fit; test rows projected). 8 SVD components.

Learners: RF canonical, RF pushed, HGB tuned; ensemble = mean of calibrated OOF.
Ablation: full vs no_E_graph.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, f1_score)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.final_push_080 import (
    log, make_A, make_B, make_C, make_H,
    learner_rf, learner_hgb, eval_pooled_oof,
    SEED, DATA_DIR, GOLD,
)

import medhg_ps.config as C, medhg_ps.data as d
from medhg_ps.data import load_raw
from medhg_ps.deploy import assemble_training_frame

OUT_LOG = Path("artifacts/newdata/final_push_080_v6.log")
OUT_RES = Path("artifacts/newdata/final_push_080_v6_results.csv")
OUT_OOF = Path("artifacts/newdata/final_push_080_v6_oof.npz")
OUT_ABL = Path("artifacts/newdata/final_push_080_v6_ablation.csv")


def _norm_id(s):
    return s.astype(str).str.replace(r"\.0+$", "", regex=True)


# ============================================================
# Precompute graph structure once (independent of fold split).
# The label-dependent parts (neighborhood outcome averages) are
# computed per fold with train-only labels.
# ============================================================
def precompute_graph(merged, orders_df, A2, A4):
    """Return dicts of neighbor sets and sparse matrices for
    fold-time feature computation."""
    all_logids = merged["LogID"].astype(str).values
    N = len(all_logids)
    lid2idx = {lid: i for i, lid in enumerate(all_logids)}

    # Provider -> set of encounter indices
    a2 = A2.copy()
    a2["LogID"] = _norm_id(a2["LogID"])
    a2["ProvID"] = _norm_id(a2["ProvID"])
    a2 = a2[a2["LogID"].isin(lid2idx)]
    a2["enc_i"] = a2["LogID"].map(lid2idx).astype(np.int64)
    prov_tokens = sorted(a2["ProvID"].unique())
    prov2id = {p: k for k, p in enumerate(prov_tokens)}
    a2["prov_i"] = a2["ProvID"].map(prov2id).astype(np.int64)

    # Sparse encounter × provider matrix (binary)
    ep_row = a2["enc_i"].values
    ep_col = a2["prov_i"].values
    ep_data = np.ones(len(ep_row), dtype=np.float32)
    M_ep = csr_matrix((ep_data, (ep_row, ep_col)),
                      shape=(N, len(prov_tokens)))
    log(f"  encounter-provider matrix {M_ep.shape} nnz={M_ep.nnz:,}")

    # Order-group co-occurrence
    orders = orders_df.copy()
    orders["LogID"] = _norm_id(orders["LogID"])
    orders["OrderGroup"] = orders["OrderGroup"].astype(str).fillna("UNK")
    orders = orders[orders["LogID"].isin(lid2idx)]
    orders["enc_i"] = orders["LogID"].map(lid2idx).astype(np.int64)
    og_tokens = sorted(orders["OrderGroup"].unique())
    og2id = {g: k for k, g in enumerate(og_tokens)}
    orders["og_i"] = orders["OrderGroup"].map(og2id).astype(np.int64)
    # aggregate counts per (encounter, order-group)
    grp = (orders.groupby(["enc_i", "og_i"]).size()
           .reset_index(name="cnt"))
    M_eo = csr_matrix((grp["cnt"].values.astype(np.float32),
                       (grp["enc_i"].values, grp["og_i"].values)),
                      shape=(N, len(og_tokens)))
    # binary presence matrix (for shared-count computations)
    M_eo_bin = (M_eo > 0).astype(np.float32)
    log(f"  encounter-ordergroup matrix {M_eo.shape} nnz={M_eo.nnz:,}")

    # Rehab / high-risk order-groups (SLP, PT, OT, Supplement, NPO, Referral, ...)
    rehab_terms = ("SLP", "PT", "OT", "Supplement", "NPO",
                   "Rehab", "Therapy", "Consult", "Referral")
    rehab_ids = [og2id[g] for g in og_tokens
                 if any(t.lower() in g.lower() for t in rehab_terms)]
    log(f"  rehab-flavored order-groups: {len(rehab_ids)}")

    # Provider volume from A4
    a4 = A4.copy()
    a4["ProvID"] = _norm_id(a4["ProvID"])
    a4["CaseVolume2yr"] = pd.to_numeric(a4["CaseVolume2yr"], errors="coerce").fillna(0)
    prov_vol = dict(zip(a4["ProvID"], a4["CaseVolume2yr"]))

    return dict(
        N=N, lid2idx=lid2idx, all_logids=all_logids,
        M_ep=M_ep, M_eo=M_eo, M_eo_bin=M_eo_bin,
        prov_tokens=prov_tokens, prov2id=prov2id,
        og_tokens=og_tokens, og2id=og2id,
        rehab_ids=rehab_ids,
        a2_prov_per_enc=a2.groupby("enc_i")["ProvID"].apply(list).to_dict(),
        prov_vol=prov_vol,
    )


# ============================================================
# Build E_graph features for one fold (train-fold labels only).
# ============================================================
def build_Egraph(gs, y, train_mask):
    """Return an (N, D) matrix of graph-structural features.
    Only y[train_mask] is used to compute neighbor outcome averages,
    so the features are leak-safe when applied to test rows."""
    N = gs["N"]
    M_ep = gs["M_ep"]      # (N, P)  encounter × provider
    M_eo = gs["M_eo"]      # (N, G)  encounter × order-group (counts)
    M_eo_bin = gs["M_eo_bin"]

    # ---- provider-shared neighborhood outcome ----
    # Shared-provider count matrix: (N, N)_sparse = M_ep @ M_ep.T
    # For each encounter e, sum over provider-sharing neighbors n of y[n],
    # weighted by number of shared providers.
    y_tr = np.zeros(N, dtype=np.float32)
    y_tr[train_mask] = y[train_mask].astype(np.float32)
    tr_mask_f = train_mask.astype(np.float32)

    # M_ep @ M_ep.T gives shared-provider counts (dense would be huge; use sparse)
    S_prov = M_ep @ M_ep.T                        # (N, N) sparse
    # For each e: numerator = sum over train neighbors n of shared_provs(e,n) * y[n]
    num_p = np.asarray(S_prov @ y_tr).ravel()
    denom_p = np.asarray(S_prov @ tr_mask_f).ravel()
    provshared_readmit = num_p / np.clip(denom_p, 1, None)

    # thresholded neighborhoods: n counted only if shares >=2 providers
    S_prov_2 = S_prov.copy()
    S_prov_2.data = (S_prov_2.data >= 2).astype(np.float32)
    S_prov_2.eliminate_zeros()
    num_p2 = np.asarray(S_prov_2 @ y_tr).ravel()
    den_p2 = np.asarray(S_prov_2 @ tr_mask_f).ravel()
    provshared2_readmit = num_p2 / np.clip(den_p2, 1, None)

    # ---- order-group-shared neighborhood outcome ----
    S_og = M_eo_bin @ M_eo_bin.T                  # (N, N) sparse
    # top-shared threshold: n counted only if shares >=5 order-groups
    for K in (3, 5, 8):
        S = S_og.copy()
        S.data = (S.data >= K).astype(np.float32)
        S.eliminate_zeros()
        if K == 3:
            og_readmit_3 = np.asarray(S @ y_tr).ravel() / np.clip(
                np.asarray(S @ tr_mask_f).ravel(), 1, None)
        elif K == 5:
            og_readmit_5 = np.asarray(S @ y_tr).ravel() / np.clip(
                np.asarray(S @ tr_mask_f).ravel(), 1, None)
        else:
            og_readmit_8 = np.asarray(S @ y_tr).ravel() / np.clip(
                np.asarray(S @ tr_mask_f).ravel(), 1, None)
    log(f"  neighbor-readmit provshared mean={provshared_readmit.mean():.3f}  "
        f"og5 mean={og_readmit_5.mean():.3f}")

    # ---- provider centrality features (label-independent, per encounter) ----
    n_prov_per_enc = np.asarray(M_ep.sum(axis=1)).ravel()  # count
    # max provider case-volume-2yr
    prov_vol_map = gs["prov_vol"]
    a2_prov_per_enc = gs["a2_prov_per_enc"]
    max_prov_vol = np.zeros(N, dtype=np.float32)
    for enc_i, provs in a2_prov_per_enc.items():
        vols = [prov_vol_map.get(p, 0) for p in provs]
        max_prov_vol[enc_i] = max(vols) if vols else 0
    # log-scale
    log_max_prov_vol = np.log1p(max_prov_vol)

    # ---- order-group diversity ----
    n_distinct_og = np.asarray(M_eo_bin.sum(axis=1)).ravel()
    # Shannon entropy per encounter
    M_eo_dense_rowsum = np.asarray(M_eo.sum(axis=1)).ravel().clip(1e-6)
    # E[-p log p] approximated row-by-row using csr
    entropy = np.zeros(N, dtype=np.float32)
    M_eo_lil = M_eo.tolil()
    for i in range(N):
        counts = np.asarray(M_eo_lil.rows[i])
        vals = np.asarray(M_eo_lil.data[i], dtype=np.float32)
        if len(vals) == 0:
            continue
        p = vals / vals.sum()
        entropy[i] = float(-(p * np.log(p + 1e-9)).sum())
    # rehab-order fraction
    rehab_ids = gs["rehab_ids"]
    rehab_count = np.asarray(M_eo_bin[:, rehab_ids].sum(axis=1)).ravel() \
                  if len(rehab_ids) else np.zeros(N)
    rehab_frac = rehab_count / np.clip(n_distinct_og, 1, None)

    # ---- SVD embedding (train-fit) of encounter × order-group ----
    svd = TruncatedSVD(n_components=8, random_state=SEED)
    M_train = M_eo_bin[train_mask]
    svd.fit(M_train)
    svd_emb = svd.transform(M_eo_bin).astype(np.float32)   # (N, 8)

    # Concatenate all features
    feats = np.column_stack([
        provshared_readmit,
        provshared2_readmit,
        og_readmit_3,
        og_readmit_5,
        og_readmit_8,
        n_prov_per_enc,
        log_max_prov_vol,
        n_distinct_og,
        entropy,
        rehab_frac,
    ]).astype(np.float32)
    feats = np.hstack([feats, svd_emb])
    log(f"  E_graph feature block: {feats.shape}")
    return feats


def main():
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    log("=== FINAL PUSH v6 (tree-based graph algorithm — path features) START ===")

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

    log("precomputing graph structure (encounter/provider/order-group)")
    from medhg_ps.data import load_order_sequence, collapse_order_runs
    orders_df = collapse_order_runs(load_order_sequence())
    raw = load_raw()
    gs = precompute_graph(merged, orders_df, raw.enc_prov_edges, raw.prov_attrs)

    log("=== CROSS VALIDATION (5-fold seed 42) ===")
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    p_oof = {"rf_base": np.full(N, np.nan),
             "rf_big":  np.full(N, np.nan),
             "hgb":     np.full(N, np.nan)}
    p_ab_no_Egraph = np.full(N, np.nan)

    for fi, (tr, te) in enumerate(skf.split(np.zeros(N), y)):
        log(f"--- fold {fi + 1}/5  train {len(tr)} test {len(te)} ---")
        train_mask = np.zeros(N, bool); train_mask[tr] = True

        XA = make_A(merged, feat_cols, cpt_arr, train_mask)
        log(f"  A {XA.shape}")

        log("  building E_graph features (leak-safe with train labels)")
        XEg = build_Egraph(gs, y, train_mask)

        X_full = np.hstack([XA, XB, XC, XH, XEg]).astype(np.float32)
        X_noE  = np.hstack([XA, XB, XC, XH]).astype(np.float32)
        if fi == 0:
            log(f"  X_full {X_full.shape}  X_noE {X_noE.shape}")

        for name, mk in [("rf_base", lambda: learner_rf()),
                         ("rf_big",  lambda: learner_rf(big=True)),
                         ("hgb",     lambda: learner_hgb())]:
            try:
                est = CalibratedClassifierCV(
                    mk(), method="isotonic", cv=3).fit(X_full[tr], y[tr])
                p_oof[name][te] = est.predict_proba(X_full[te])[:, 1]
                log(f"  fold {fi + 1} {name} done")
            except Exception as e:
                log(f"  fold {fi + 1} {name} FAILED: {e}")

        try:
            est = CalibratedClassifierCV(
                learner_rf(), method="isotonic", cv=3).fit(X_noE[tr], y[tr])
            p_ab_no_Egraph[te] = est.predict_proba(X_noE[te])[:, 1]
        except Exception as e:
            log(f"  ablation FAILED: {e}")

    log("=== POOLED OOF EVAL ===")
    results = []
    for name in ["rf_base", "rf_big", "hgb"]:
        p = p_oof[name]
        if np.isnan(p).any():
            log(f"  {name} has NaN"); continue
        r = eval_pooled_oof(y, p, name)
        log(f"  {name:10s} AUROC {r['auroc']:.3f} "
            f"({r['auroc_lo']:.3f}-{r['auroc_hi']:.3f}) "
            f"AUPRC {r['auprc']:.3f} F1 {r['f1']:.3f}")
        results.append(r)

    valid = [k for k in p_oof if not np.isnan(p_oof[k]).any()]
    if len(valid) >= 2:
        p_ens = np.mean([p_oof[k] for k in valid], axis=0)
        r = eval_pooled_oof(y, p_ens, "ensemble_v6")
        log(f"  ensemble_v6 AUROC {r['auroc']:.3f} "
            f"({r['auroc_lo']:.3f}-{r['auroc_hi']:.3f}) "
            f"AUPRC {r['auprc']:.3f} F1 {r['f1']:.3f}")
        results.append(r)

    log("=== ABLATION (rf_base only) ===")
    ablation = []
    for tag, p in [("rf_base_full",       p_oof["rf_base"]),
                   ("rf_base_no_Egraph",  p_ab_no_Egraph)]:
        if np.isnan(p).any():
            log(f"  {tag} has NaN"); continue
        au = roc_auc_score(y, p); ap = average_precision_score(y, p)
        ablation.append(dict(config=tag, auroc=au, auprc=ap))
        log(f"  {tag:22s} AUROC {au:.3f}  AUPRC {ap:.3f}")

    pd.DataFrame(results).to_csv(OUT_RES, index=False)
    pd.DataFrame(ablation).to_csv(OUT_ABL, index=False)
    np.savez(OUT_OOF, y=y, p_ab_no_Egraph=p_ab_no_Egraph, **p_oof)
    log(f"saved {OUT_RES}, {OUT_ABL}, {OUT_OOF}")
    log("=== DONE ===")


if __name__ == "__main__":
    main()
