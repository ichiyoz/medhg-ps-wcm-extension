"""Final push v9b — Flow-first, lean tabular design.

Hypothesis: patient flow through the hospital carries signal. Orders are
proxies for clinical decisions. Rather than duplicating information across
tabular + graph, put everything decision-related into the flow (GRU on
timestamped order sequence + static graph context), and reserve the
tabular block for information the flow cannot reconstruct (demographics,
chronic conditions, prior-utilization, SDoH).

Tabular block (LEAN):
  Age, Sex, PatientType, ASA           (demographics + acuity type)
  17 Charlson comorbidity flags         (chronic conditions)
  ED180d, admits365d                    (utilization history)
  Geocode / ACS (6 features)            (SDoH)
  ~28 columns total

Dropped from tabular:
  HCT, Na, Cr, Alb, WBC labs  →  captured in graph via PROC:Lab orders
  CPT one-hot                 →  static context concatenated to GRU
  LACE / HOSPITAL scores      →  redundant with graph flow
  Care-path Fseq features     →  redundant with GRU
  Notes, SDoH regex           →  outside this experiment's scope

Sequence GRU input (per encounter, per step):
  order_group_id  (81 categories)
  time_bucket_id  (6 buckets: 0-1h, 1-4h, 4-12h, 12-24h, 24-48h, 48h+)
  order_source_id (MED vs PROC)
  Embedded, concatenated, run through 64-hidden GRU
  Extract final hidden state → 64-d

Static graph context concatenated to GRU output:
  Top-40 CPT one-hot          (procedure that drove this encounter)
  Provider role mix (5 counts) (n_attending, n_resident, n_crna, n_anesth, n_other)
  n_orders, bundle_fraction   (sequence-level scalars)

Total feature matrix per encounter:
  ~28 tabular + 64 GRU + 40 CPT + 5 role + 2 seq = ~139 columns

Learners: rf_big, XGBoost (tuning params), LightGBM (tuning params)
Ablation: full, no-gru, no-cpt, no-role, tabular-only
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, f1_score)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.final_push_080 import (
    log, learner_rf, eval_pooled_oof, SEED, DATA_DIR, GOLD,
)
from analysis.final_push_080_v8 import PROV_ROLES, prov_role, _norm_id

import medhg_ps.config as C, medhg_ps.data as d
from medhg_ps.data import (load_raw, load_order_sequence, collapse_order_runs,
                            fit_preprocess, apply_preprocess)
from medhg_ps.deploy import assemble_training_frame

OUT_LOG = Path("artifacts/newdata/final_push_080_v9b.log")
OUT_RES = Path("artifacts/newdata/final_push_080_v9b_results.csv")
OUT_OOF = Path("artifacts/newdata/final_push_080_v9b_oof.npz")
OUT_ABL = Path("artifacts/newdata/final_push_080_v9b_ablation.csv")

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
GRU_HIDDEN = 64
GRU_ORDER_EMB = 32
GRU_TIME_EMB = 8
GRU_SOURCE_EMB = 4
MAXLEN = 128
TOP_CPT = 40


# ============================================================
# Tabular block: LEAN — only non-overlapping with GRU flow.
# Dropped: PatientType (admission type = HOSPITAL-P, inferrable from flow),
#          ASAClass (pre-op rating, overlaps with acuity captured by flow),
#          Race (potential bias source per NIH guidance).
# ============================================================
DEMO_COLS = ["AgeYears", "Gender"]
UTIL_COLS = ["n_ed_visits_180d", "n_admits_365d"]


def make_tabular_lean(merged, train_mask):
    """Assemble lean tabular block: demographics + Charlson + utilization
    + geocode. Fit preprocessing on train fold only, apply to all rows."""
    # demographics
    demo = merged[DEMO_COLS].copy()
    st_demo = fit_preprocess(demo.loc[train_mask].reset_index(drop=True),
                              id_cols=[])[1]
    X_demo = apply_preprocess(demo, st_demo)

    # Charlson (from LACE_Comp.csv)
    lace_path = Path(DATA_DIR) / "LACE_Comp.csv"
    lace = pd.read_csv(lace_path, header=None, encoding="utf-8-sig")
    lace.columns = ["LogID"] + [f"cci_{i}" for i in range(1, 18)] + \
                   ["ed180d", "admits365d"]
    lace["LogID"] = _norm_id(lace["LogID"])
    joined = pd.DataFrame({"LogID": merged["LogID"].astype(str).values}) \
                .merge(lace, on="LogID", how="left")
    dx_cols = [f"cci_{i}" for i in range(1, 18)]
    X_cci = joined[dx_cols].fillna(0).values.astype(np.float32)
    X_util = joined[["ed180d", "admits365d"]].fillna(0).values.astype(np.float32)
    X_util = np.log1p(X_util)  # log-scaled

    # Geocode — reuse make_H from v6
    from analysis.final_push_080 import make_H
    X_geo, _ = make_H(merged)
    X_geo = X_geo.astype(np.float32)

    X_tab = np.hstack([X_demo, X_cci, X_util, X_geo]).astype(np.float32)
    return X_tab


# ============================================================
# Static graph context
# ============================================================
DISPO_GROUP = {
    "Home or Self Care": "home",
    "Home-Health Care Svc": "home_health",
    "Skilled Nursing Facility": "snf",
    "Rehab Facility": "rehab",
    "Long Term Care": "rehab",
    "Interim / Custodial Care Facility": "rehab",
    "Hospice/Home": "hospice_exp",
    "Hospice/Medical Facility": "hospice_exp",
    "Expired": "hospice_exp",
    "Expired in Medical Facility": "hospice_exp",
    "Acute / Short Term Hospital": "acute_transfer",
    "Psychiatric Hospital": "acute_transfer",
    "Cancer Center/Children's Hospital": "acute_transfer",
    "Left Against Medical Advice": "ama",
    "Another Health Care Institution Not Defined": "other",
}
DISPO_ORDER = ["home", "home_health", "snf", "rehab", "hospice_exp",
               "acute_transfer", "ama", "other"]


def make_static_context(merged, cpt_arr, orders_df, A2, A4, train_mask):
    """Return static per-encounter context: top-40 CPT one-hot,
    provider role mix, sequence-length scalar, bundle-fraction scalar,
    discharge-disposition one-hot (8 categories)."""
    N = len(merged)
    all_logids = _norm_id(merged["LogID"]).values

    # CPT one-hot (top 40)
    # cpt_arr may be (N,) or (N, 1); flatten to strings
    cpt_flat = np.asarray(cpt_arr).ravel().astype(str)
    # Determine top CPTs from train fold to avoid leakage
    tr_cpt = cpt_flat[train_mask]
    unique, counts = np.unique(tr_cpt, return_counts=True)
    top_idx = np.argsort(-counts)[:TOP_CPT]
    top_cpts = unique[top_idx]
    cpt2j = {str(c): j for j, c in enumerate(top_cpts)}
    X_cpt = np.zeros((N, TOP_CPT + 1), dtype=np.float32)  # +1 for "other"
    for i, c in enumerate(cpt_flat):
        if c in cpt2j:
            X_cpt[i, cpt2j[c]] = 1.0
        else:
            X_cpt[i, TOP_CPT] = 1.0

    # Provider role mix (per-encounter counts)
    a2 = A2.copy()
    a2["LogID"] = _norm_id(a2["LogID"])
    a2["ProvID"] = _norm_id(a2["ProvID"])
    a4 = A4.copy()
    a4["ProvID"] = _norm_id(a4["ProvID"])
    a4["role"] = a4["ProvType"].apply(prov_role)
    a2 = a2.merge(a4[["ProvID", "role"]], on="ProvID", how="left")
    a2["role"] = a2["role"].fillna("other")

    lid2idx = {lid: i for i, lid in enumerate(all_logids)}
    a2["enc_i"] = a2["LogID"].map(lid2idx)
    a2 = a2.dropna(subset=["enc_i"])
    a2["enc_i"] = a2["enc_i"].astype(int)

    X_role = np.zeros((N, len(PROV_ROLES)), dtype=np.float32)
    for role in PROV_ROLES:
        counts_by_enc = a2[a2["role"] == role].groupby("enc_i").size()
        X_role[counts_by_enc.index.values, PROV_ROLES.index(role)] = \
            counts_by_enc.values.astype(np.float32)

    # Sequence-length + bundle-fraction scalars
    orders_ts = orders_df.copy()
    orders_ts["LogID"] = _norm_id(orders_ts["LogID"])
    orders_ts = orders_ts[orders_ts["LogID"].isin(lid2idx)]
    ts_col = "RunStart" if "RunStart" in orders_ts.columns else "OrderTime"
    orders_ts[ts_col] = pd.to_datetime(orders_ts[ts_col], errors="coerce")
    orders_ts = orders_ts.dropna(subset=[ts_col])

    orders_ts["min_bucket"] = orders_ts[ts_col].dt.floor("2min")
    bundle = (orders_ts.groupby(["LogID", "min_bucket"])
              .size().rename("sz").reset_index())
    orders_ts = orders_ts.merge(bundle, on=["LogID", "min_bucket"], how="left")
    orders_ts["is_bundled"] = (orders_ts["sz"] >= 3).astype(int)

    n_orders = orders_ts.groupby("LogID").size().rename("n").reset_index()
    n_orders["enc_i"] = n_orders["LogID"].map(lid2idx).astype(int)
    X_n_orders = np.zeros(N, dtype=np.float32)
    X_n_orders[n_orders["enc_i"].values] = np.log1p(n_orders["n"].values)

    bfrac = orders_ts.groupby("LogID")["is_bundled"].mean().rename("f").reset_index()
    bfrac["enc_i"] = bfrac["LogID"].map(lid2idx).astype(int)
    X_bfrac = np.zeros(N, dtype=np.float32)
    X_bfrac[bfrac["enc_i"].values] = bfrac["f"].values.astype(np.float32)

    X_seq_scalar = np.column_stack([X_n_orders, X_bfrac]).astype(np.float32)

    # Discharge Disposition -- binary: home vs other.
    # (Home = "Home or Self Care" + "Home-Health Care Svc"; per user
    # directive to collapse home-health into home.)
    HOME_LABELS = {"Home or Self Care", "Home-Health Care Svc"}
    dispo_raw = merged["Discharge Disposition"].astype(str).fillna("other")
    X_dispo = dispo_raw.isin(HOME_LABELS).astype(np.float32).values.reshape(-1, 1)

    return dict(X_cpt=X_cpt, X_role=X_role, X_seq=X_seq_scalar,
                X_dispo=X_dispo,
                orders_ts=orders_ts, lid2idx=lid2idx)


# ============================================================
# Multi-attribute GRU: reads (order_group, time_bucket, source) triples
# ============================================================
class MultiAttrSeqGRU(nn.Module):
    def __init__(self, n_orders, n_time, n_source):
        super().__init__()
        self.emb_o = nn.Embedding(n_orders + 1, GRU_ORDER_EMB, padding_idx=0)
        self.emb_t = nn.Embedding(n_time + 1, GRU_TIME_EMB, padding_idx=0)
        self.emb_s = nn.Embedding(n_source + 1, GRU_SOURCE_EMB, padding_idx=0)
        in_dim = GRU_ORDER_EMB + GRU_TIME_EMB + GRU_SOURCE_EMB
        self.gru = nn.GRU(in_dim, GRU_HIDDEN, batch_first=True)
        self.clf = nn.Linear(GRU_HIDDEN, 1)

    def forward(self, order_ids, time_ids, src_ids, lens):
        # order_ids: (B, T)  int64
        eo = self.emb_o(order_ids)
        et = self.emb_t(time_ids)
        es = self.emb_s(src_ids)
        x = torch.cat([eo, et, es], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        h = h.squeeze(0)  # (B, hidden)
        logits = self.clf(h).squeeze(-1)
        return logits, h


def bucketize_time(minutes_from_admit):
    """Return bucket 1..6 for 0-60, 60-240, 240-720, 720-1440, 1440-2880, 2880+.
    Uses 0 as padding."""
    x = np.asarray(minutes_from_admit, dtype=np.float32)
    b = np.ones(x.shape, dtype=np.int64)
    b[x < 60] = 1
    b[(x >= 60) & (x < 240)] = 2
    b[(x >= 240) & (x < 720)] = 3
    b[(x >= 720) & (x < 1440)] = 4
    b[(x >= 1440) & (x < 2880)] = 5
    b[x >= 2880] = 6
    return b


def build_sequences(orders_ts, lid2idx, N):
    """Return three padded numpy arrays: order_ids (N, MAXLEN),
    time_ids (N, MAXLEN), src_ids (N, MAXLEN), lens (N,)."""
    orders_ts = orders_ts.copy()
    orders_ts["OrderGroup"] = orders_ts["OrderGroup"].astype(str).fillna("UNK")
    orders_ts["OrderSource"] = orders_ts["OrderSource"].astype(str).fillna("UNK")
    og_tokens = sorted(orders_ts["OrderGroup"].unique())
    og2id = {t: i + 1 for i, t in enumerate(og_tokens)}      # 0 = padding
    src_tokens = sorted(orders_ts["OrderSource"].unique())
    src2id = {t: i + 1 for i, t in enumerate(src_tokens)}    # 0 = padding
    n_orders = len(og_tokens)
    n_source = len(src_tokens)

    # Per-encounter t=0 = first order time
    ts_col = "RunStart" if "RunStart" in orders_ts.columns else "OrderTime"
    orders_ts[ts_col] = pd.to_datetime(orders_ts[ts_col], errors="coerce")
    t_start = orders_ts.groupby("LogID")[ts_col].transform("min")
    orders_ts["mins_from_admit"] = (
        (orders_ts[ts_col] - t_start).dt.total_seconds() / 60.0
    )
    orders_ts = orders_ts.sort_values(["LogID", "SeqInEncounter"])
    orders_ts["og_id"] = orders_ts["OrderGroup"].map(og2id).astype(np.int64)
    orders_ts["src_id"] = orders_ts["OrderSource"].map(src2id).astype(np.int64)
    orders_ts["time_id"] = bucketize_time(orders_ts["mins_from_admit"].values)

    order_seqs = np.zeros((N, MAXLEN), dtype=np.int64)
    time_seqs  = np.zeros((N, MAXLEN), dtype=np.int64)
    src_seqs   = np.zeros((N, MAXLEN), dtype=np.int64)
    lens = np.ones(N, dtype=np.int64)   # min length 1 for pack_padded_sequence

    for lid, grp in orders_ts.groupby("LogID"):
        if lid not in lid2idx:
            continue
        i = lid2idx[lid]
        seq_o = grp["og_id"].values[:MAXLEN]
        seq_t = grp["time_id"].values[:MAXLEN]
        seq_s = grp["src_id"].values[:MAXLEN]
        L = len(seq_o)
        order_seqs[i, :L] = seq_o
        time_seqs[i, :L] = seq_t
        src_seqs[i, :L] = seq_s
        lens[i] = max(1, L)     # avoid zero-len for GRU pack

    return order_seqs, time_seqs, src_seqs, lens, n_orders, n_source


def train_gru_and_encode(order_seqs, time_seqs, src_seqs, lens,
                         y, tr, n_orders, n_source, epochs=5, seed=SEED):
    """Train GRU on train indices, extract encoder state for all rows."""
    torch.manual_seed(seed)
    # BUGFIX: previous version passed n_source twice, sizing the time embedding
    # from the source vocab (=2) even though time buckets range 1..6. This
    # produced out-of-range index accesses that PyTorch handled silently on MPS,
    # yielding garbage embeddings for later time buckets. Fixed by sizing the
    # time embedding from the actual bucket range in the data.
    n_time = int(np.asarray(time_seqs).max())
    model = MultiAttrSeqGRU(n_orders, n_time, n_source).to(DEV)
    # class weights
    y_tr = y[tr]
    pos_w = torch.tensor([(y_tr == 0).sum() / max(1, (y_tr == 1).sum())],
                         dtype=torch.float32, device=DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    O_t = torch.tensor(order_seqs, dtype=torch.int64, device=DEV)
    T_t = torch.tensor(time_seqs, dtype=torch.int64, device=DEV)
    S_t = torch.tensor(src_seqs, dtype=torch.int64, device=DEV)
    L_t = torch.tensor(lens, dtype=torch.int64, device=DEV)
    y_all = torch.tensor(y, dtype=torch.float32, device=DEV)

    tr_idx = np.asarray(tr)
    for ep in range(epochs):
        model.train()
        # shuffle
        rng = np.random.default_rng(seed + ep)
        perm = rng.permutation(len(tr_idx))
        batch_size = 256
        for bs in range(0, len(perm), batch_size):
            ii = tr_idx[perm[bs:bs + batch_size]]
            oi, ti, si, li = O_t[ii], T_t[ii], S_t[ii], L_t[ii]
            logits, _ = model(oi, ti, si, li)
            loss = F.binary_cross_entropy_with_logits(
                logits, y_all[ii], pos_weight=pos_w)
            opt.zero_grad(); loss.backward(); opt.step()

    # extract encoder state for all rows
    model.eval()
    embs = np.zeros((len(y), GRU_HIDDEN), dtype=np.float32)
    with torch.no_grad():
        for bs in range(0, len(y), 512):
            ii = np.arange(bs, min(bs + 512, len(y)))
            oi, ti, si, li = O_t[ii], T_t[ii], S_t[ii], L_t[ii]
            _, h = model(oi, ti, si, li)
            embs[ii] = h.detach().cpu().numpy()
    return embs


def main():
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    log("=== FINAL PUSH v9b (lean tabular + flow GRU + graph context) START ===")

    merged, feat_cols, cpt_arr, Fseq, seq_all, _ = assemble_training_frame()
    gold = pd.read_parquet(GOLD)[["LogID", "ReadmittedWithin30Days_gold"]]
    gold["LogID"] = gold["LogID"].astype(str)
    merged["LogID"] = merged["LogID"].astype(str)
    merged = merged.merge(gold, on="LogID", how="left")

    # ---- COHORT EXCLUSION ----
    # Standard readmission-modeling exclusions:
    #   - Expired / Expired in Medical Facility: no discharge, y undefined
    #   - Hospice/Home + Hospice/Medical Facility: limited life expectancy
    #   - Acute / Short Term Hospital: transfer, readmission tracked elsewhere
    #   - Left Against Medical Advice: different clinical trajectory
    EXCLUDE_LABELS = {
        "Expired",
        "Expired in Medical Facility",
        "Hospice/Home",
        "Hospice/Medical Facility",
        "Acute / Short Term Hospital",
        "Left Against Medical Advice",
    }
    n_before = len(merged)
    excl_mask = merged["Discharge Disposition"].astype(str).isin(EXCLUDE_LABELS)
    dropped = merged.loc[excl_mask, "Discharge Disposition"].value_counts()
    keep_mask = ~excl_mask
    merged = merged.loc[keep_mask].reset_index(drop=True)
    cpt_arr = np.asarray(cpt_arr)[keep_mask.values]
    log(f"cohort exclusion: dropped {int(excl_mask.sum())} patients "
        f"({n_before} -> {len(merged)})")
    for label, n in dropped.items():
        log(f"  - {label}: {n}")

    y = merged["ReadmittedWithin30Days_gold"].astype(int).values
    N = len(y)
    log(f"cohort N={N} base rate {y.mean()*100:.2f}%")

    # Load static substrate
    orders_df = collapse_order_runs(load_order_sequence())
    raw = load_raw()

    # Build sequences ONCE (structure doesn't need train-fold info)
    log("building flow sequences from orders")
    all_logids = _norm_id(merged["LogID"]).values
    lid2idx = {lid: i for i, lid in enumerate(all_logids)}
    order_seqs, time_seqs, src_seqs, lens, n_ord_vocab, n_src_vocab = \
        build_sequences(orders_df, lid2idx, N)
    log(f"  order-vocab {n_ord_vocab}  source-vocab {n_src_vocab}  "
        f"median len {int(np.median(lens))}  max len {int(lens.max())}")

    log("=== CROSS VALIDATION (5-fold seed 42) ===")
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    p_oof = {"rf_big": np.full(N, np.nan),
             "xgb":    np.full(N, np.nan),
             "lgbm":   np.full(N, np.nan)}
    # Ablations on rf_big
    p_ab_no_gru = np.full(N, np.nan)
    p_ab_no_cpt = np.full(N, np.nan)
    p_ab_no_role = np.full(N, np.nan)
    p_ab_no_dispo = np.full(N, np.nan)
    p_ab_tab_only = np.full(N, np.nan)

    for fi, (tr, te) in enumerate(skf.split(np.zeros(N), y)):
        log(f"--- fold {fi + 1}/5  train {len(tr)} test {len(te)} ---")
        train_mask = np.zeros(N, bool); train_mask[tr] = True

        log("  building tabular (lean)")
        X_tab = make_tabular_lean(merged, train_mask)
        log(f"  X_tab {X_tab.shape}")

        log("  building static context (CPT + role mix + seq scalars)")
        ctx = make_static_context(merged, cpt_arr, orders_df,
                                   raw.enc_prov_edges, raw.prov_attrs,
                                   train_mask)
        log(f"  X_cpt {ctx['X_cpt'].shape}  X_role {ctx['X_role'].shape}  "
            f"X_seq {ctx['X_seq'].shape}  X_dispo {ctx['X_dispo'].shape}")

        log("  training multi-attr flow GRU")
        t0 = time.time()
        X_gru = train_gru_and_encode(
            order_seqs, time_seqs, src_seqs, lens, y, tr,
            n_ord_vocab, n_src_vocab, epochs=5, seed=SEED + fi)
        log(f"  X_gru {X_gru.shape} in {time.time()-t0:.1f}s")

        X_full = np.hstack([X_tab, X_gru, ctx["X_cpt"], ctx["X_role"],
                            ctx["X_seq"], ctx["X_dispo"]]).astype(np.float32)
        X_no_gru  = np.hstack([X_tab,        ctx["X_cpt"], ctx["X_role"],
                               ctx["X_seq"], ctx["X_dispo"]]).astype(np.float32)
        X_no_cpt  = np.hstack([X_tab, X_gru,               ctx["X_role"],
                               ctx["X_seq"], ctx["X_dispo"]]).astype(np.float32)
        X_no_role = np.hstack([X_tab, X_gru, ctx["X_cpt"],
                               ctx["X_seq"], ctx["X_dispo"]]).astype(np.float32)
        X_no_dispo = np.hstack([X_tab, X_gru, ctx["X_cpt"], ctx["X_role"],
                                ctx["X_seq"]]).astype(np.float32)
        X_tab_only = X_tab

        if fi == 0:
            log(f"  X_full {X_full.shape}  X_tab_only {X_tab_only.shape}")

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

        # Ablation on rf_big
        for Xab, arr, tag in [(X_no_gru,   p_ab_no_gru,   "no_gru"),
                              (X_no_cpt,   p_ab_no_cpt,   "no_cpt"),
                              (X_no_role,  p_ab_no_role,  "no_role"),
                              (X_no_dispo, p_ab_no_dispo, "no_dispo"),
                              (X_tab_only, p_ab_tab_only, "tab_only")]:
            try:
                est = CalibratedClassifierCV(
                    learner_rf(big=True), method="isotonic", cv=3).fit(
                    Xab[tr], y[tr])
                arr[te] = est.predict_proba(Xab[te])[:, 1]
                log(f"  ablation {tag} done")
            except Exception as e:
                log(f"  ablation {tag} FAILED: {e}")

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
        r = eval_pooled_oof(y, p_ens, "ensemble_v9b")
        log(f"  ensemble_v9b AUROC {r['auroc']:.3f} "
            f"({r['auroc_lo']:.3f}-{r['auroc_hi']:.3f}) "
            f"AUPRC {r['auprc']:.3f} F1 {r['f1']:.3f}")
        results.append(r)

    log("=== ABLATION (rf_big only) ===")
    ablation = []
    for tag, p in [("rf_big_full",     p_oof["rf_big"]),
                   ("rf_big_no_gru",   p_ab_no_gru),
                   ("rf_big_no_cpt",   p_ab_no_cpt),
                   ("rf_big_no_role",  p_ab_no_role),
                   ("rf_big_no_dispo", p_ab_no_dispo),
                   ("rf_big_tab_only", p_ab_tab_only)]:
        if np.isnan(p).any():
            log(f"  {tag} has NaN skip"); continue
        au = roc_auc_score(y, p); ap = average_precision_score(y, p)
        ablation.append(dict(config=tag, auroc=au, auprc=ap))
        log(f"  {tag:20s} AUROC {au:.3f}  AUPRC {ap:.3f}")

    pd.DataFrame(results).to_csv(OUT_RES, index=False)
    pd.DataFrame(ablation).to_csv(OUT_ABL, index=False)
    np.savez(OUT_OOF, y=y,
             p_ab_no_gru=p_ab_no_gru,
             p_ab_no_cpt=p_ab_no_cpt,
             p_ab_no_role=p_ab_no_role,
             p_ab_no_dispo=p_ab_no_dispo,
             p_ab_tab_only=p_ab_tab_only,
             **p_oof)
    log(f"saved {OUT_RES}, {OUT_ABL}, {OUT_OOF}")
    log("=== DONE ===")


if __name__ == "__main__":
    main()
