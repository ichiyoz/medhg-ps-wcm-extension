"""Final push v5 — v4 with proper clock-time normalization + order-set handling.

Fixes two issues with v4:
1. v4 used SeqInEncounter (1, 2, 3, ...) for the time proxy. Orders that fire
   from the same order set share a clock time but get spread across sequence
   positions, so bundled orders artificially span time. v5 uses OrderTime
   (actual clock time), so bundled orders collapse to the same norm_pos.
2. v4 gave equal weight to bundled and deliberate orders. Bundled orders are
   less independent clinical decisions -- a 12-item post-op recovery set
   carries less information per token than 12 individually-placed orders.
   v5 adds a bundle_penalty = 1 / bundle_size and multiplies it into the
   edge weight.

Block set (same as v4):
  A base tabular + CPT one-hot
  B LACE utility (Charlson + ED180d + admits365d)
  C LACE + HOSPITAL scores + components
  H geocode + ACS
  E_temp temporal-edge graph embedding (64-d, NOW clock-time based)

Also adds a small block:
  X_bundle_frac : per-encounter fraction of orders that arrived in bundles
                  (single scalar concatenated into A). This captures
                  "routine order-set-driven stay" vs "actively-managed stay".
"""
from __future__ import annotations
import os, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import dgl, dgl.function as fn
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_auc_score, average_precision_score)
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.final_push_080 import (
    log, make_A, make_B, make_C, make_H,
    learner_rf, learner_hgb, eval_pooled_oof,
    SEED, DATA_DIR, GOLD,
)
# Reuse v4's GNN classes
from analysis.final_push_080_v4 import (
    WeightedHeteroConv, TemporalHeteroGNN, train_temporal_gnn,
    VAL_FRAC, EMB_DIM, GNN_DEV,
)

import medhg_ps.config as C, medhg_ps.data as d
from medhg_ps.data import fit_preprocess, apply_preprocess, load_raw
from medhg_ps.deploy import assemble_training_frame

OUT_LOG = Path("artifacts/newdata/final_push_080_v5.log")
OUT_RES = Path("artifacts/newdata/final_push_080_v5_results.csv")
OUT_OOF = Path("artifacts/newdata/final_push_080_v5_oof.npz")
OUT_ABL = Path("artifacts/newdata/final_push_080_v5_ablation.csv")

BUNDLE_WINDOW_MIN = 2      # orders within 2 min of each other = bundled
TOP_K_PER_ENC = 30


# ============================================================
# Clock-time edge computation with order-set awareness
# ============================================================
def build_temporal_hg_v5(merged, orders_df, A2, A4):
    """Same shape as v4's build_temporal_hg, but:
      - norm_pos derived from OrderTime, not SeqInEncounter
      - each order carries a bundle_penalty = 1/bundle_size
      - edge_weight = norm_pos * mean(bundle_penalty) per (LogID, OrderGroup)
      - also returns per-encounter bundle_frac feature (fraction of orders
        that arrived in a bundle).
    """
    all_logids = merged["LogID"].astype(str).values
    enc2id = {lid: i for i, lid in enumerate(all_logids)}
    N_enc = len(all_logids)

    orders = orders_df.copy()
    orders["LogID"] = orders["LogID"].astype(str)
    orders["OrderGroup"] = orders["OrderGroup"].astype(str).fillna("UNK")
    # collapse_order_runs renames OrderTime -> RunStart/RunEnd. Use RunStart
    # as the clock-time timestamp for each collapsed run.
    ts_col = "RunStart" if "RunStart" in orders.columns else "OrderTime"
    orders[ts_col] = pd.to_datetime(orders[ts_col], errors="coerce")
    orders = orders.dropna(subset=[ts_col])

    # --- CLOCK-TIME norm_pos ---
    per_enc = orders.groupby("LogID")[ts_col].agg(t_start="min", t_end="max")
    orders = orders.merge(per_enc.reset_index(), on="LogID", how="left")
    dur_s = (orders["t_end"] - orders["t_start"]).dt.total_seconds().clip(lower=1.0)
    from_start_s = (orders[ts_col] - orders["t_start"]).dt.total_seconds()
    orders["norm_pos"] = (from_start_s / dur_s).clip(0.0, 1.0)

    # --- BUNDLE DETECTION ---
    # Count orders sharing the same LogID + minute bucket as a bundle.
    orders["min_bucket"] = orders[ts_col].dt.floor(f"{BUNDLE_WINDOW_MIN}min")
    bundle = (orders.groupby(["LogID", "min_bucket"])
              .size().rename("bundle_size").reset_index())
    orders = orders.merge(bundle, on=["LogID", "min_bucket"], how="left")
    orders["bundle_penalty"] = 1.0 / orders["bundle_size"].clip(lower=1.0)
    orders["is_bundled"] = (orders["bundle_size"] >= 3).astype(int)  # 3+ per 2-min
    n_bundled = int(orders["is_bundled"].sum())
    log(f"  bundle stats: {n_bundled:,}/{len(orders):,} orders in bundles (>=3 per 2min)  "
        f"median bundle size {int(orders['bundle_size'].median())} "
        f"p95 {int(orders['bundle_size'].quantile(0.95))} "
        f"max {int(orders['bundle_size'].max())}")

    # --- Aggregate to (LogID, OrderGroup) with clock-time-weighted edges ---
    grouped = (orders.groupby(["LogID", "OrderGroup"])
               .agg(count=("norm_pos", "size"),
                    mean_pos=("norm_pos", "mean"),
                    mean_bpen=("bundle_penalty", "mean"))
               .reset_index())
    grouped["rk"] = grouped.groupby("LogID")["count"].rank(ascending=False, method="first")
    grouped = grouped[grouped["rk"] <= TOP_K_PER_ENC]
    grouped = grouped[grouped["LogID"].isin(enc2id)]
    grouped["enc_id"] = grouped["LogID"].map(enc2id).astype(np.int64)

    og_tokens = sorted(grouped["OrderGroup"].unique())
    og2id = {t: i for i, t in enumerate(og_tokens)}
    grouped["og_id"] = grouped["OrderGroup"].map(og2id).astype(np.int64)
    N_og = len(og_tokens)

    # Edge weight combines clock-time position AND inverse-bundle-size.
    e_weight = (grouped["mean_pos"] * grouped["mean_bpen"]).values.astype(np.float32)
    log(f"  edges enc->og: {len(grouped):,}  order-groups: {N_og}  "
        f"weight median {np.median(e_weight):.3f}  p25 {np.percentile(e_weight,25):.3f} "
        f"p75 {np.percentile(e_weight,75):.3f}")

    enc_src_og = torch.tensor(grouped["enc_id"].values, dtype=torch.int64)
    og_dst = torch.tensor(grouped["og_id"].values, dtype=torch.int64)
    ew = torch.tensor(e_weight, dtype=torch.float32)

    # --- PROVIDER nodes + edges (unweighted, same as v4) ---
    a2 = A2.copy()
    a2["LogID"] = a2["LogID"].astype(str)
    a2["ProvID"] = a2["ProvID"].astype(str).str.replace(r"\.0+$", "", regex=True)
    a4 = A4.copy()
    a4["ProvID"] = a4["ProvID"].astype(str).str.replace(r"\.0+$", "", regex=True)
    a2 = a2[a2["LogID"].isin(enc2id) & a2["ProvID"].isin(a4["ProvID"])]
    prov_tokens = sorted(a2["ProvID"].unique())
    prov2id = {p: i for i, p in enumerate(prov_tokens)}
    a2["enc_id"] = a2["LogID"].map(enc2id).astype(np.int64)
    a2["prov_id"] = a2["ProvID"].map(prov2id).astype(np.int64)
    N_prov = len(prov_tokens)
    log(f"  edges enc->prov: {len(a2):,}  providers: {N_prov}")

    enc_src_prov = torch.tensor(a2["enc_id"].values, dtype=torch.int64)
    prov_dst = torch.tensor(a2["prov_id"].values, dtype=torch.int64)

    data_dict = {
        ("encounter", "ordered", "order_group"): (enc_src_og, og_dst),
        ("order_group", "ordered_by", "encounter"): (og_dst, enc_src_og),
        ("encounter", "treated_by", "provider"): (enc_src_prov, prov_dst),
        ("provider", "treats", "encounter"): (prov_dst, enc_src_prov),
    }
    num_nodes = {"encounter": N_enc, "order_group": N_og, "provider": N_prov}
    hg = dgl.heterograph(data_dict, num_nodes_dict=num_nodes)

    hg.edges["ordered"].data["w"] = ew
    hg.edges["ordered_by"].data["w"] = ew
    hg.edges["treated_by"].data["w"] = torch.ones(len(a2), dtype=torch.float32)
    hg.edges["treats"].data["w"] = torch.ones(len(a2), dtype=torch.float32)

    # Node features
    og_feats = np.eye(N_og, dtype=np.float32)
    a4_sub = a4[a4["ProvID"].isin(prov_tokens)].set_index("ProvID").loc[prov_tokens]
    ptype = np.asarray(a4_sub["ProvType"].fillna("Unknown").astype(str).values, dtype=object)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    prov_feats = ohe.fit_transform(ptype.reshape(-1, 1)).astype(np.float32)
    log(f"  provider feature dim: {prov_feats.shape[1]}")

    # --- Per-encounter bundle_frac scalar feature for the tabular block ---
    bundle_frac = (orders.groupby("LogID")["is_bundled"].mean()
                   .rename("bundle_frac").reset_index())
    bundle_frac["LogID"] = bundle_frac["LogID"].astype(str)
    bundle_frac_map = dict(zip(bundle_frac["LogID"], bundle_frac["bundle_frac"]))
    bundle_frac_arr = np.array(
        [bundle_frac_map.get(lid, 0.0) for lid in all_logids],
        dtype=np.float32,
    ).reshape(-1, 1)
    log(f"  bundle_frac stats: mean {float(bundle_frac_arr.mean()):.3f}  "
        f"median {float(np.median(bundle_frac_arr)):.3f}  "
        f"max {float(bundle_frac_arr.max()):.3f}")

    return hg, {"order_group": og_feats, "provider": prov_feats}, enc2id, bundle_frac_arr


def main():
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    log("=== FINAL PUSH v5 (clock-time temporal edges + order-set aware) START ===")

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

    log("BLOCK E_temp: build clock-time-weighted heterograph + bundle features")
    from medhg_ps.data import load_order_sequence, collapse_order_runs
    orders_df = collapse_order_runs(load_order_sequence())
    raw = load_raw()
    hg, static_feats, enc2id, X_bundle_frac = build_temporal_hg_v5(
        merged, orders_df, raw.enc_prov_edges, raw.prov_attrs,
    )
    log(f"  hg ntypes {hg.ntypes}  etypes {hg.etypes}")

    log("=== CROSS VALIDATION (5-fold seed 42) ===")
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    p_oof = {"rf_base": np.full(N, np.nan),
             "rf_big":  np.full(N, np.nan),
             "hgb":     np.full(N, np.nan)}
    p_ab_no_Etemp = np.full(N, np.nan)

    for fi, (tr, te) in enumerate(skf.split(np.zeros(N), y)):
        log(f"--- fold {fi + 1}/5  train {len(tr)} test {len(te)} ---")
        train_mask = np.zeros(N, bool); train_mask[tr] = True

        # A + bundle_frac scalar
        XA_base = make_A(merged, feat_cols, cpt_arr, train_mask)
        XA = np.hstack([XA_base, X_bundle_frac]).astype(np.float32)
        log(f"  A + bundle_frac  {XA.shape}")

        rng = np.random.default_rng(SEED + fi)
        tr_perm = rng.permutation(tr)
        n_val = int(round(VAL_FRAC * N))
        val = tr_perm[:n_val]
        tr_only = tr_perm[n_val:]

        feat_all = merged[feat_cols].copy()
        _, st = fit_preprocess(feat_all.loc[tr_only].reset_index(drop=True), id_cols=[])
        X_enc = apply_preprocess(feat_all, st).astype(np.float32)

        log("  train clock-time temporal-edge GNN")
        try:
            XE_temp = train_temporal_gnn(
                hg, static_feats, X_enc, y, tr_only, val, N,
                seed=SEED + fi,
            )
        except Exception as e:
            log(f"  E_temp FAILED: {e}; falling back to zeros")
            XE_temp = np.zeros((N, EMB_DIM), dtype=np.float32)

        X_full = np.hstack([XA, XB, XC, XH, XE_temp]).astype(np.float32)
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
            p_ab_no_Etemp[te] = est.predict_proba(X_noE[te])[:, 1]
        except Exception as e:
            log(f"  ablation FAILED: {e}")

    log("=== POOLED OOF EVAL ===")
    results = []
    for name in ["rf_base", "rf_big", "hgb"]:
        p = p_oof[name]
        if np.isnan(p).any():
            log(f"  {name} has NaN, skipping"); continue
        r = eval_pooled_oof(y, p, name)
        log(f"  {name:10s} AUROC {r['auroc']:.3f} "
            f"({r['auroc_lo']:.3f}-{r['auroc_hi']:.3f}) "
            f"AUPRC {r['auprc']:.3f} F1 {r['f1']:.3f}")
        results.append(r)

    valid = [k for k in p_oof if not np.isnan(p_oof[k]).any()]
    if len(valid) >= 2:
        p_ens = np.mean([p_oof[k] for k in valid], axis=0)
        r = eval_pooled_oof(y, p_ens, "ensemble_v5")
        log(f"  ensemble_v5 AUROC {r['auroc']:.3f} "
            f"({r['auroc_lo']:.3f}-{r['auroc_hi']:.3f}) "
            f"AUPRC {r['auprc']:.3f} F1 {r['f1']:.3f}")
        results.append(r)

    log("=== ABLATION (rf_base only) ===")
    ablation = []
    for tag, p in [("rf_base_full",     p_oof["rf_base"]),
                   ("rf_base_no_Etemp", p_ab_no_Etemp)]:
        if np.isnan(p).any():
            log(f"  {tag} has NaN skipping"); continue
        au = roc_auc_score(y, p); ap = average_precision_score(y, p)
        ablation.append(dict(config=tag, auroc=au, auprc=ap))
        log(f"  {tag:20s} AUROC {au:.3f}  AUPRC {ap:.3f}")

    pd.DataFrame(results).to_csv(OUT_RES, index=False)
    pd.DataFrame(ablation).to_csv(OUT_ABL, index=False)
    np.savez(OUT_OOF, y=y, p_ab_no_Etemp=p_ab_no_Etemp, **p_oof)
    log(f"saved {OUT_RES}, {OUT_ABL}, {OUT_OOF}")
    log("=== DONE ===")


if __name__ == "__main__":
    main()
