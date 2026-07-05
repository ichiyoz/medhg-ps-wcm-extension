"""Final push v8 — Rich node/edge graph design.

Architectural extension of v6/v7. Same tree-based-graph paradigm
(pre-compute graph features, feed to trees) but with a much richer
graph substrate:

Nodes:
  encounter            (14,009)
  provider_attending   \\
  provider_resident    |
  provider_crna        |  Split provider by role (5 subtypes)
  provider_anesth      |
  provider_other       /
  order_med            \\
  order_lab            |  Split orders by category (5 subtypes)
  order_imaging        |
  order_consult        |
  order_care           /
  diagnosis            (17 Charlson comorbidity categories from LACE_Comp)
  order_set            (bundle IDs from 2-min-window detection)

Edges (all bipartite encounter↔type, weighted by count):
  encounter → each provider-role
  encounter → each order-category
  encounter → diagnosis node
  encounter → order_set

Features derived from this graph:
  E_svd  TruncatedSVD-16 of the concatenated encounter×[all-node-types]
         matrix — captures spectral factorization across all edge types
  E_lp   Label-propagation neighborhood readmit rate, computed
         separately per edge-type family:
           - share ≥2 attendings
           - share ≥2 residents
           - share ≥3 medications categories
           - share ≥3 lab types
           - share ≥1 imaging
           - share ≥1 diagnosis
           - share ≥1 order-set
         (7 features)
  E_deg  Degree features: number of distinct attendings, residents, CRNAs,
         med orders, lab orders, imaging orders, consults, diagnoses,
         order-sets per encounter (9 features)
  E_n2v  Node2Vec 64-d walk embedding on the full multigraph
         (encounter + all node types combined)

Learners: tuned LightGBM (best from tuning experiment) + tuned RF,
         plus untuned XGBoost. Ensemble = mean of three.
Ablation: full vs no_E_svd vs no_E_lp vs no_E_deg vs no_E_n2v vs no_graphs
"""
from __future__ import annotations
import json, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from scipy.sparse import csr_matrix, hstack as sparse_hstack
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, f1_score)
import xgboost as xgb
import lightgbm as lgb
from gensim.models import Word2Vec

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.final_push_080 import (
    log, make_A, make_B, make_C, make_H,
    learner_rf, learner_hgb, eval_pooled_oof,
    SEED, DATA_DIR, GOLD,
)

import medhg_ps.config as C, medhg_ps.data as d
from medhg_ps.data import load_raw, load_order_sequence, collapse_order_runs
from medhg_ps.deploy import assemble_training_frame

OUT_LOG = Path("artifacts/newdata/final_push_080_v8.log")
OUT_RES = Path("artifacts/newdata/final_push_080_v8_results.csv")
OUT_OOF = Path("artifacts/newdata/final_push_080_v8_oof.npz")
OUT_ABL = Path("artifacts/newdata/final_push_080_v8_ablation.csv")

SEED = 42

# Provider role mapping
PROV_ROLE_MAP = {
    "Attending": "attending",
    "Physician": "attending",
    "Resident": "resident",
    "Fellow": "resident",
    "Medical Student": "resident",
    "Nurse Anesthetist": "crna",
    "Student Nurse Anesthetist": "crna",
    "Anesthesiologist": "anesth",
    "Physician Assistant": "other",
    "Technician": "other",
    "Perfusionist": "other",
    "Dentist": "other",
}
PROV_ROLES = ["attending", "resident", "crna", "anesth", "other"]

# Order-group categorization
LAB_KEYWORDS = ("Lab", "Point of Care", "Microbiology",
                "Pathology", "Blood Bank")
IMAGING_KEYWORDS = ("Imaging", "EKG")
CONSULT_KEYWORDS = ("Consult",)


def _norm_id(s):
    return s.astype(str).str.replace(r"\.0+$", "", regex=True)


def categorize_ordergroup(og: str) -> str:
    """Return one of: 'med', 'lab', 'imaging', 'consult', 'care'."""
    if og.startswith("MED:"):
        return "med"
    body = og.split(":", 1)[1] if ":" in og else og
    for kw in LAB_KEYWORDS:
        if kw.lower() in body.lower():
            return "lab"
    for kw in IMAGING_KEYWORDS:
        if kw.lower() in body.lower():
            return "imaging"
    for kw in CONSULT_KEYWORDS:
        if kw.lower() in body.lower():
            return "consult"
    return "care"


def prov_role(ptype):
    return PROV_ROLE_MAP.get(str(ptype), "other")


def build_rich_graph_substrate(merged, orders_df, A2, A4):
    """Build sparse encounter×feature matrices for each node type family.
    Returns dict of csr_matrix and mapping metadata."""
    all_logids = _norm_id(merged["LogID"]).values
    N = len(all_logids)
    lid2idx = {lid: i for i, lid in enumerate(all_logids)}

    # -------- providers by role --------
    a2 = A2.copy()
    a2["LogID"] = _norm_id(a2["LogID"])
    a2["ProvID"] = _norm_id(a2["ProvID"])
    a2 = a2[a2["LogID"].isin(lid2idx)]
    a2["enc_i"] = a2["LogID"].map(lid2idx).astype(np.int64)

    a4 = A4.copy()
    a4["ProvID"] = _norm_id(a4["ProvID"])
    a4["role"] = a4["ProvType"].apply(prov_role)

    a2 = a2.merge(a4[["ProvID", "role"]], on="ProvID", how="left")
    a2["role"] = a2["role"].fillna("other")

    prov_mats = {}
    for role in PROV_ROLES:
        sub = a2[a2["role"] == role]
        if len(sub) == 0:
            prov_mats[role] = csr_matrix((N, 1), dtype=np.float32)
            continue
        tokens = sorted(sub["ProvID"].unique())
        tok2j = {t: j for j, t in enumerate(tokens)}
        sub_j = sub["ProvID"].map(tok2j).values
        M = csr_matrix((np.ones(len(sub), dtype=np.float32),
                        (sub["enc_i"].values, sub_j)),
                       shape=(N, len(tokens)))
        prov_mats[role] = M
        log(f"  provider role '{role}': {len(tokens)} nodes, "
            f"{M.nnz:,} edges")

    # -------- orders by category --------
    orders = orders_df.copy()
    orders["LogID"] = _norm_id(orders["LogID"])
    orders["OrderGroup"] = orders["OrderGroup"].astype(str).fillna("UNK")
    orders = orders[orders["LogID"].isin(lid2idx)]
    orders["enc_i"] = orders["LogID"].map(lid2idx).astype(np.int64)
    orders["og_cat"] = orders["OrderGroup"].apply(categorize_ordergroup)

    order_mats = {}
    for cat in ["med", "lab", "imaging", "consult", "care"]:
        sub = orders[orders["og_cat"] == cat]
        if len(sub) == 0:
            order_mats[cat] = csr_matrix((N, 1), dtype=np.float32)
            continue
        tokens = sorted(sub["OrderGroup"].unique())
        tok2j = {t: j for j, t in enumerate(tokens)}
        sub_j = sub["OrderGroup"].map(tok2j).values
        # count per (enc, og)
        counts = (sub.groupby(["enc_i", "OrderGroup"]).size()
                  .reset_index(name="cnt"))
        counts["j"] = counts["OrderGroup"].map(tok2j).values
        M = csr_matrix((counts["cnt"].values.astype(np.float32),
                        (counts["enc_i"].values, counts["j"].values)),
                       shape=(N, len(tokens)))
        order_mats[cat] = M
        log(f"  order category '{cat}': {len(tokens)} nodes, "
            f"{M.nnz:,} edges")

    # -------- diagnosis nodes (Charlson) --------
    lace_path = Path(DATA_DIR) / "LACE_Comp.csv"
    lace = pd.read_csv(lace_path, header=None, encoding="utf-8-sig")
    # Column 0 = LogID; columns 1..17 = 17 Charlson flags; col 18 = ED180d;
    # col 19 = admits365d (per LACE_Components.sql schema)
    lace.columns = ["LogID"] + [f"cci_{i}" for i in range(1, 18)] + \
                    ["ed180d", "admits365d"]
    lace["LogID"] = _norm_id(lace["LogID"])
    lace_idx = pd.DataFrame({"LogID": all_logids}).merge(lace, on="LogID",
                                                         how="left")
    dx_cols = [f"cci_{i}" for i in range(1, 18)]
    lace_idx[dx_cols] = lace_idx[dx_cols].fillna(0).astype(np.float32)
    M_dx = csr_matrix(lace_idx[dx_cols].values)
    log(f"  diagnosis (Charlson): {len(dx_cols)} nodes, "
        f"{M_dx.nnz:,} edges (positive Charlson flags)")

    # -------- order-set nodes (2-min-window bundles) --------
    orders_ts = orders.copy()
    # collapse_order_runs renames OrderTime -> RunStart / RunEnd
    ts_col = "RunStart" if "RunStart" in orders_ts.columns else "OrderTime"
    orders_ts[ts_col] = pd.to_datetime(orders_ts[ts_col], errors="coerce")
    orders_ts = orders_ts.dropna(subset=[ts_col])
    orders_ts["min_bucket"] = orders_ts[ts_col].dt.floor("2min")
    bundle_size = (orders_ts.groupby(["LogID", "min_bucket"])
                   .size().rename("sz").reset_index())
    bundle_size = bundle_size[bundle_size["sz"] >= 3]        # at least 3 orders
    bundle_size["bundle_key"] = (
        bundle_size["LogID"].astype(str) + "::" +
        bundle_size["min_bucket"].astype(str))
    # Bundle IDENTITY = the set of order-groups fired together. Get that.
    bundle_orders = (orders_ts.merge(bundle_size[["LogID", "min_bucket"]],
                                     on=["LogID", "min_bucket"], how="inner")
                     .groupby(["LogID", "min_bucket"])["OrderGroup"]
                     .apply(lambda s: "|".join(sorted(set(s))))
                     .rename("signature").reset_index())
    bundle_orders["sig_idx"] = bundle_orders.groupby("signature").ngroup()
    n_bundle_sigs = int(bundle_orders["sig_idx"].max() + 1) if \
                    len(bundle_orders) else 0
    log(f"  order-set signatures: {n_bundle_sigs} unique bundles "
        f"across {len(bundle_orders):,} bundle events")
    # Cap to top 200 most frequent bundles (long tail of one-offs is noise)
    top_bundles = (bundle_orders["sig_idx"].value_counts().head(200).index
                   .tolist())
    b2j = {b: j for j, b in enumerate(top_bundles)}
    bundle_orders["j"] = bundle_orders["sig_idx"].map(b2j)
    bundle_orders = bundle_orders.dropna(subset=["j"])
    bundle_orders["enc_i"] = bundle_orders["LogID"].astype(str).map(lid2idx)
    bundle_orders = bundle_orders.dropna(subset=["enc_i"])
    if len(bundle_orders) > 0:
        M_bundle = csr_matrix(
            (np.ones(len(bundle_orders), dtype=np.float32),
             (bundle_orders["enc_i"].astype(int).values,
              bundle_orders["j"].astype(int).values)),
            shape=(N, len(top_bundles)))
    else:
        M_bundle = csr_matrix((N, 1), dtype=np.float32)
    log(f"  bundle nodes (top-200 by frequency): {M_bundle.shape[1]}, "
        f"{M_bundle.nnz:,} edges")

    return dict(
        N=N, lid2idx=lid2idx, all_logids=all_logids,
        prov_mats=prov_mats,
        order_mats=order_mats,
        M_dx=M_dx,
        M_bundle=M_bundle,
    )


def _neighbor_readmit(M_source_bin, y, train_mask, threshold):
    """Label propagation: for each row, mean outcome of TRAIN rows
    sharing ≥threshold columns with this row."""
    N = M_source_bin.shape[0]
    y_tr = np.zeros(N, dtype=np.float32); y_tr[train_mask] = y[train_mask]
    tr_mask_f = train_mask.astype(np.float32)
    S = M_source_bin @ M_source_bin.T
    S.data = (S.data >= threshold).astype(np.float32)
    S.eliminate_zeros()
    num = np.asarray(S @ y_tr).ravel()
    den = np.asarray(S @ tr_mask_f).ravel()
    return num / np.clip(den, 1, None)


def build_Egraph_v8(gs, y, train_mask):
    """Compute E_graph feature block from the rich graph substrate.
    Returns (N, D) tuple(features, group_dims) where group_dims tags
    which features come from which sub-family for the ablation."""
    N = gs["N"]

    # ---- E_deg: degree features ----
    deg_feats = []
    for role in PROV_ROLES:
        M = gs["prov_mats"][role]
        deg_feats.append(np.asarray(M.sum(axis=1)).ravel())
    for cat in ["med", "lab", "imaging", "consult", "care"]:
        M = (gs["order_mats"][cat] > 0).astype(np.float32)
        deg_feats.append(np.asarray(M.sum(axis=1)).ravel())
    deg_feats.append(np.asarray(gs["M_dx"].sum(axis=1)).ravel())
    deg_feats.append(np.asarray(gs["M_bundle"].sum(axis=1)).ravel())
    E_deg = np.column_stack(deg_feats).astype(np.float32)
    log(f"  E_deg {E_deg.shape}")

    # ---- E_lp: label-propagation neighborhood readmit rates ----
    thresholds = {
        "attending": 2,
        "resident": 2,
        "med": 3,
        "lab": 3,
        "imaging": 1,
        "consult": 1,
        "diagnosis": 1,
        "bundle": 1,
    }
    lp_feats = []
    lp_names = []
    for role in ["attending", "resident"]:
        M_bin = (gs["prov_mats"][role] > 0).astype(np.float32)
        lp_feats.append(_neighbor_readmit(M_bin, y, train_mask, thresholds[role]))
        lp_names.append(f"lp_{role}")
    for cat in ["med", "lab", "imaging", "consult"]:
        M_bin = (gs["order_mats"][cat] > 0).astype(np.float32)
        lp_feats.append(_neighbor_readmit(M_bin, y, train_mask, thresholds[cat]))
        lp_names.append(f"lp_order_{cat}")
    M_dx_bin = (gs["M_dx"] > 0).astype(np.float32)
    lp_feats.append(_neighbor_readmit(M_dx_bin, y, train_mask, thresholds["diagnosis"]))
    lp_names.append("lp_dx")
    M_bundle_bin = (gs["M_bundle"] > 0).astype(np.float32)
    lp_feats.append(_neighbor_readmit(M_bundle_bin, y, train_mask, thresholds["bundle"]))
    lp_names.append("lp_bundle")
    E_lp = np.column_stack(lp_feats).astype(np.float32)
    log(f"  E_lp {E_lp.shape}")

    # ---- E_svd: TruncatedSVD-16 on concat of all encounter×type matrices ----
    parts = []
    for role in PROV_ROLES:
        parts.append((gs["prov_mats"][role] > 0).astype(np.float32))
    for cat in ["med", "lab", "imaging", "consult", "care"]:
        parts.append(gs["order_mats"][cat])
    parts.append(gs["M_dx"])
    parts.append((gs["M_bundle"] > 0).astype(np.float32))
    M_all = sparse_hstack(parts, format="csr").astype(np.float32)
    log(f"  concatenated matrix {M_all.shape} nnz={M_all.nnz:,}")
    svd = TruncatedSVD(n_components=16, random_state=SEED).fit(M_all[train_mask])
    E_svd = svd.transform(M_all).astype(np.float32)
    log(f"  E_svd {E_svd.shape} explained-var {svd.explained_variance_ratio_.sum():.3f}")

    return dict(
        E_deg=E_deg,   # (N, 12)
        E_lp=E_lp,     # (N, 8)
        E_svd=E_svd,   # (N, 16)
    )


# ============================================================
# Fast Node2Vec on the rich multi-node graph
# ============================================================
def compute_n2v_v8(gs, dim=64, walk_len=15, num_walks=8, window=5, seed=SEED):
    """Node2Vec walks over the multi-type graph. All node types collapsed
    into a single graph with unique IDs."""
    N_enc = gs["N"]
    node_id = N_enc  # running ID after encounters
    edges = []   # list of (i, j) undirected

    # encounter <-> provider role subtypes
    for role in PROV_ROLES:
        M = gs["prov_mats"][role]
        coo = M.tocoo()
        for i, j in zip(coo.row, coo.col):
            edges.append((int(i), node_id + int(j)))
        node_id += M.shape[1]

    # encounter <-> order categories
    for cat in ["med", "lab", "imaging", "consult", "care"]:
        M = (gs["order_mats"][cat] > 0)
        coo = M.tocoo()
        for i, j in zip(coo.row, coo.col):
            edges.append((int(i), node_id + int(j)))
        node_id += M.shape[1]

    # encounter <-> diagnosis
    coo = gs["M_dx"].tocoo()
    for i, j in zip(coo.row, coo.col):
        edges.append((int(i), node_id + int(j)))
    node_id += gs["M_dx"].shape[1]

    # encounter <-> order-set
    coo = gs["M_bundle"].tocoo()
    for i, j in zip(coo.row, coo.col):
        edges.append((int(i), node_id + int(j)))
    node_id += gs["M_bundle"].shape[1]

    num_nodes = node_id
    log(f"  multigraph: {num_nodes:,} nodes, {len(edges):,} directed edges")

    # CSR-style adjacency
    edges_arr = np.array(edges, dtype=np.int64)
    rows = np.concatenate([edges_arr[:, 0], edges_arr[:, 1]])
    cols = np.concatenate([edges_arr[:, 1], edges_arr[:, 0]])
    order = np.argsort(rows)
    rows_s = rows[order]; cols_s = cols[order]
    indptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.add.at(indptr, rows_s + 1, 1)
    np.cumsum(indptr, out=indptr)
    indices = cols_s.astype(np.int64)

    log(f"  generating walks (walks={num_walks}, walk_len={walk_len})")
    rng = np.random.default_rng(seed)
    walks = []
    starts = np.arange(num_nodes, dtype=np.int64)
    for _ in range(num_walks):
        rng.shuffle(starts)
        for s in starts:
            walk = [s]; cur = s
            for _step in range(walk_len - 1):
                lo, hi = indptr[cur], indptr[cur + 1]
                if hi == lo:
                    break
                cur = int(indices[rng.integers(lo, hi)])
                walk.append(cur)
            walks.append([str(w) for w in walk])
    log(f"  {len(walks):,} walks generated")

    log(f"  fitting Word2Vec skip-gram (dim={dim}, window={window})")
    t0 = time.time()
    w2v = Word2Vec(sentences=walks, vector_size=dim, window=window,
                   min_count=1, sg=1, workers=4, seed=seed, epochs=5)
    log(f"  Word2Vec fit in {time.time()-t0:.1f}s  vocab {len(w2v.wv.key_to_index):,}")

    emb = np.zeros((N_enc, dim), dtype=np.float32)
    for i in range(N_enc):
        key = str(i)
        if key in w2v.wv:
            emb[i] = w2v.wv[key]
    log(f"  encounter embedding {emb.shape}")
    return emb


def main():
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    log("=== FINAL PUSH v8 (rich node/edge design) START ===")

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

    log("building rich graph substrate")
    orders_df = collapse_order_runs(load_order_sequence())
    raw = load_raw()
    gs = build_rich_graph_substrate(merged, orders_df,
                                    raw.enc_prov_edges, raw.prov_attrs)

    # Node2Vec on multi-node graph — computed ONCE (transductive)
    log("computing Node2Vec on multi-type multigraph")
    t0 = time.time()
    X_n2v = compute_n2v_v8(gs)
    log(f"Node2Vec built in {time.time()-t0:.1f}s")

    log("=== CROSS VALIDATION (5-fold seed 42) ===")
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    p_oof = {"rf_big": np.full(N, np.nan),
             "xgb":    np.full(N, np.nan),
             "lgbm":   np.full(N, np.nan)}
    p_ab_no_deg  = np.full(N, np.nan)
    p_ab_no_lp   = np.full(N, np.nan)
    p_ab_no_svd  = np.full(N, np.nan)
    p_ab_no_n2v  = np.full(N, np.nan)
    p_ab_no_all  = np.full(N, np.nan)

    for fi, (tr, te) in enumerate(skf.split(np.zeros(N), y)):
        log(f"--- fold {fi + 1}/5  train {len(tr)} test {len(te)} ---")
        train_mask = np.zeros(N, bool); train_mask[tr] = True

        XA = make_A(merged, feat_cols, cpt_arr, train_mask)
        log(f"  A {XA.shape}")

        log("  computing E_graph_v8 features")
        E = build_Egraph_v8(gs, y, train_mask)

        X_base = np.hstack([XA, XB, XC, XH]).astype(np.float32)
        X_full = np.hstack([X_base, E["E_deg"], E["E_lp"], E["E_svd"],
                            X_n2v]).astype(np.float32)
        # Ablation
        X_no_deg = np.hstack([X_base, E["E_lp"], E["E_svd"], X_n2v]).astype(np.float32)
        X_no_lp  = np.hstack([X_base, E["E_deg"], E["E_svd"], X_n2v]).astype(np.float32)
        X_no_svd = np.hstack([X_base, E["E_deg"], E["E_lp"], X_n2v]).astype(np.float32)
        X_no_n2v = np.hstack([X_base, E["E_deg"], E["E_lp"], E["E_svd"]]).astype(np.float32)
        X_no_all = X_base

        if fi == 0:
            log(f"  X_full {X_full.shape} X_base {X_base.shape}")

        # Fit 3 learners on X_full
        for name, mk in [
            ("rf_big", lambda: learner_rf(big=True)),
            ("xgb",    lambda: xgb.XGBClassifier(
                n_estimators=300, learning_rate=0.02, max_depth=5,
                min_child_weight=8, subsample=0.75, colsample_bytree=0.6,
                reg_lambda=0.3, gamma=1.5, tree_method="hist",
                eval_metric="logloss", random_state=SEED,
                n_jobs=-1, verbosity=0, use_label_encoder=False)),
            ("lgbm",   lambda: lgb.LGBMClassifier(
                n_estimators=500, learning_rate=0.015, num_leaves=127,
                max_depth=3, min_child_samples=20, subsample=0.75,
                colsample_bytree=0.6, reg_lambda=1.0, class_weight=None,
                random_state=SEED, n_jobs=-1, verbosity=-1)),
        ]:
            try:
                est = CalibratedClassifierCV(mk(), method="isotonic",
                                             cv=3).fit(X_full[tr], y[tr])
                p_oof[name][te] = est.predict_proba(X_full[te])[:, 1]
                log(f"  fold {fi + 1} {name} done")
            except Exception as e:
                log(f"  fold {fi + 1} {name} FAILED: {e}")

        # Ablation on rf_big (main representative)
        for Xab, arr, tag in [(X_no_deg, p_ab_no_deg, "no_deg"),
                              (X_no_lp,  p_ab_no_lp,  "no_lp"),
                              (X_no_svd, p_ab_no_svd, "no_svd"),
                              (X_no_n2v, p_ab_no_n2v, "no_n2v"),
                              (X_no_all, p_ab_no_all, "no_graph")]:
            try:
                est = CalibratedClassifierCV(
                    learner_rf(big=True), method="isotonic", cv=3).fit(Xab[tr], y[tr])
                arr[te] = est.predict_proba(Xab[te])[:, 1]
                log(f"  ablation {tag} done")
            except Exception as e:
                log(f"  ablation {tag} FAILED: {e}")

    # Eval
    log("=== POOLED OOF EVAL ===")
    results = []
    for name in ["rf_big", "xgb", "lgbm"]:
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
        r = eval_pooled_oof(y, p_ens, "ensemble_v8")
        log(f"  ensemble_v8 AUROC {r['auroc']:.3f} "
            f"({r['auroc_lo']:.3f}-{r['auroc_hi']:.3f}) "
            f"AUPRC {r['auprc']:.3f} F1 {r['f1']:.3f}")
        results.append(r)

    log("=== ABLATION (rf_big only) ===")
    ablation = []
    for tag, p in [("rf_big_full",     p_oof["rf_big"]),
                   ("rf_big_no_deg",   p_ab_no_deg),
                   ("rf_big_no_lp",    p_ab_no_lp),
                   ("rf_big_no_svd",   p_ab_no_svd),
                   ("rf_big_no_n2v",   p_ab_no_n2v),
                   ("rf_big_no_graph", p_ab_no_all)]:
        if np.isnan(p).any():
            log(f"  {tag} has NaN skip"); continue
        au = roc_auc_score(y, p); ap = average_precision_score(y, p)
        ablation.append(dict(config=tag, auroc=au, auprc=ap))
        log(f"  {tag:20s} AUROC {au:.3f}  AUPRC {ap:.3f}")

    pd.DataFrame(results).to_csv(OUT_RES, index=False)
    pd.DataFrame(ablation).to_csv(OUT_ABL, index=False)
    np.savez(OUT_OOF, y=y,
             p_ab_no_deg=p_ab_no_deg,
             p_ab_no_lp=p_ab_no_lp,
             p_ab_no_svd=p_ab_no_svd,
             p_ab_no_n2v=p_ab_no_n2v,
             p_ab_no_all=p_ab_no_all,
             **p_oof)
    log(f"saved {OUT_RES}, {OUT_ABL}, {OUT_OOF}")
    log("=== DONE ===")


if __name__ == "__main__":
    main()
