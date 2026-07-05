"""Final push v3 — MedHG-PS with ORDER-GROUPS as third node type + LACE/HOSPITAL
variables baked in. Drops F (BERT) and G+I (regex from notes) per user directive.
Keeps H (geocode).

Block set (v3):
  A base tabular (~200)
  B LACE utility (Charlson x 17 + ED180d + admits365d + interactions, 25)
  C LACE + HOSPITAL scores + components (16)
  D_order order-sequence GRU embedding (64)
  D_unit Fseq care-path tabular (33)
  E1_v3 MedHG-PS ie-HGCN on encounter + provider + ORDER-GROUP heterograph (~96)
  E2 care-unit-sequence GRU embedding (64)
  H geocode + ACS (6)

E1_v3 replaces v2's E1 by swapping A3 unit nodes for order-group nodes in the
ie-HGCN. Reuses medhg_ps.graph.build_graph by monkey-patching the raw bundle
at fold time so unit_ids -> order-group tokens and enc_unit_edges -> encounter
to order-group edges (weighted by count of that order in the encounter).
"""
from __future__ import annotations
import os, sys, time, warnings
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.final_push_080 import (
    log, make_A, make_B, make_C, make_H, SeqGRU, train_gru_and_encode,
    learner_rf, learner_hgb, eval_pooled_oof,
    SEED, DATA_DIR, GOLD,
)
# Reuse v2's E1 GNN pipeline and E2 unit-seq builder
from analysis.final_push_080_v2 import (
    train_gnn_and_encode, build_unit_seqs, GNN_CFG, GNN_DEV, VAL_FRAC, _norm,
)

import medhg_ps.config as C, medhg_ps.data as d
from medhg_ps.data import load_raw, build_provider_features

OUT_LOG = Path("artifacts/newdata/final_push_080_v3.log")
OUT_RES = Path("artifacts/newdata/final_push_080_v3_results.csv")
OUT_OOF = Path("artifacts/newdata/final_push_080_v3_oof.npz")
OUT_ABL = Path("artifacts/newdata/final_push_080_v3_ablation.csv")


# ---------------------------------------------------------------------
# Swap A3 units for ORDER-GROUPS in the raw bundle used by build_graph.
# We construct a synthetic "unit" node type where each token is an
# OrderGroup and each encounter connects to each of its distinct
# OrderGroups via the enc_unit_edges DataFrame.
# ---------------------------------------------------------------------
def build_order_group_substrate(raw, merged, orders_df):
    """Return (new_raw, ordergroup_ids_df, X_ordergroup) with A3 replaced by
    encounter -> order-group edges, where order-groups are ~80 tokens."""
    orders = orders_df.copy()
    orders["LogID"] = _norm(orders["LogID"])
    orders["OrderGroup"] = orders["OrderGroup"].astype(str).fillna("UNK")

    # per (LogID, OrderGroup) count; used as edge weight / provenance
    edge_df = (orders.groupby(["LogID", "OrderGroup"])
               .size().rename("count").reset_index())
    # limit to top 30 order-groups per encounter to bound density
    edge_df["rank"] = edge_df.groupby("LogID")["count"].rank(
        ascending=False, method="first")
    edge_df = edge_df[edge_df["rank"] <= 30]

    # Order-group vocab as "unit" nodes; assign DepartmentID = OrderGroup token
    og_tokens = sorted(edge_df["OrderGroup"].unique())
    og_ids = pd.DataFrame({"DepartmentID": og_tokens})
    og_ids["DepartmentName"] = og_tokens

    # Learned-embedding-init features: one-hot per token (cap size)
    n_og = len(og_tokens)
    X_og = np.eye(n_og, dtype=np.float32)                # simple one-hot
    log(f"  order-group substrate: {n_og} tokens, "
        f"{len(edge_df):,} enc<->og edges "
        f"(median {int(edge_df.groupby('LogID').size().median())} og per enc)")

    # Build synthetic enc_unit_edges: LogID + DepartmentID (+ Hours=count)
    synth = edge_df.rename(columns={"OrderGroup": "DepartmentID"}).copy()
    synth["Hours"] = synth["count"].astype(float)
    # add other cols that A3 has so downstream code doesn't break
    for c in ("EncounterCSN", "PAT_ID", "DepartmentName", "UnitType",
              "InstitutionType", "InTime", "OutTime", "SeqInEncounter"):
        if c not in synth.columns:
            synth[c] = np.nan

    new_raw = copy(raw)
    new_raw.enc_unit_edges = synth[
        ["LogID", "EncounterCSN", "PAT_ID", "DepartmentID", "DepartmentName",
         "UnitType", "InstitutionType", "InTime", "OutTime", "Hours",
         "SeqInEncounter"]
    ]
    new_raw.unit_attrs = og_ids                          # minimal — will only need IDs
    return new_raw, og_ids, X_og


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    log("=== FINAL PUSH v3 (MedHG-PS with orders as A3; drop F, G+I) START ===")

    # cohort + gold label
    from medhg_ps.deploy import assemble_training_frame
    merged, feat_cols, cpt_arr, Fseq, seq_all, _ = assemble_training_frame()
    gold = pd.read_parquet(GOLD)[["LogID", "ReadmittedWithin30Days_gold"]]
    gold["LogID"] = gold["LogID"].astype(str)
    merged["LogID"] = merged["LogID"].astype(str)
    merged = merged.merge(gold, on="LogID", how="left")
    y = merged["ReadmittedWithin30Days_gold"].astype(int).values
    N = len(y)
    log(f"cohort N={N} base rate {y.mean()*100:.2f}%")

    log("BLOCK B: LACE utility"); XB, colsB = make_B(merged); log(f"  B {XB.shape}")
    log("BLOCK C: LACE + HOSPITAL scores"); XC, colsC = make_C(merged); log(f"  C {XC.shape}")
    log("BLOCK H: geocode + ACS"); XH, colsH = make_H(merged); log(f"  H {XH.shape}")

    # D_order: order sequences
    log("BLOCK D_order: order-seq loader")
    from medhg_ps.data import load_order_sequence, collapse_order_runs
    orders_df = collapse_order_runs(load_order_sequence())
    orders_df["LogID"] = orders_df["LogID"].astype(str)
    tokens = orders_df["OrderGroup"].astype(str).fillna("UNK")
    o_vocab = {t: i + 1 for i, t in enumerate(sorted(tokens.unique()))}
    orders_df["tid"] = tokens.map(o_vocab).astype(np.int64)
    all_logids = merged["LogID"].astype(str).values
    o_seqs_by = (orders_df.sort_values(["LogID", "SeqInEncounter"])
                 .groupby("LogID")["tid"].apply(list))
    order_seqs = [np.asarray(o_seqs_by.get(lid, []), dtype=np.int64)
                  for lid in all_logids]
    o_vocab_size = len(o_vocab) + 2
    log(f"  order-seq vocab={o_vocab_size} median_len="
        f"{int(np.median([len(s) for s in order_seqs]))}")

    # D_unit: care-path tabular
    Fseq["LogID"] = Fseq["LogID"].astype(str)
    care_df = pd.DataFrame({"LogID": all_logids}).merge(Fseq, on="LogID", how="left")
    care_cols = [c for c in care_df.columns if c != "LogID"]
    XCARE = care_df[care_cols].fillna(0).values.astype(np.float32)
    log(f"BLOCK D_unit care-path tabular {XCARE.shape}")

    # Graph substrate: PROVIDERS from A2/A4, ORDER-GROUPS as third node type
    log("loading raw graph tables for E1 (MedHG-PS with orders-as-A3)")
    raw = load_raw()
    prov_ids, X_prov, _ = build_provider_features(raw.prov_attrs)
    raw_og, og_ids, X_og = build_order_group_substrate(raw, merged, orders_df)

    # A3 unit sequence for E2 (kept)
    log("A3 unit trajectory for E2 (unit-seq GRU)")
    a3 = d._read_table(C.UNIT_EDGES_PARQUET, C.A3_UNIT_EDGES_COLUMNS)
    unit_seqs, u_vocab_size = build_unit_seqs(merged, a3)

    # 5-fold CV
    log("=== CROSS VALIDATION (5-fold seed 42) ===")
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    p_oof = {"rf_base": np.full(N, np.nan),
             "rf_big":  np.full(N, np.nan),
             "hgb":     np.full(N, np.nan)}
    p_ab_no_E1  = np.full(N, np.nan)
    p_ab_no_E2  = np.full(N, np.nan)
    p_ab_no_E12 = np.full(N, np.nan)

    for fi, (tr, te) in enumerate(skf.split(np.zeros(N), y)):
        log(f"--- fold {fi + 1}/5  train {len(tr)} test {len(te)} ---")
        train_mask = np.zeros(N, bool); train_mask[tr] = True

        XA = make_A(merged, feat_cols, cpt_arr, train_mask)
        log(f"  A {XA.shape}")

        log("  train order-GRU (5 epochs)")
        XD_o = train_gru_and_encode(order_seqs, y, tr, o_vocab_size,
                                    maxlen=128, epochs=5, hidden=64)

        log("  train MedHG-PS ie-HGCN with ORDER-GROUP substrate")
        try:
            XE1 = train_gnn_and_encode(merged, feat_cols, cpt_arr, y, tr,
                                       raw_og, prov_ids, X_prov,
                                       og_ids, X_og,
                                       fold_seed=SEED + fi)
            log(f"  E1_v3 emb {XE1.shape}")
        except Exception as e:
            log(f"  E1_v3 FAILED: {e}; falling back to zero-vector 96-d")
            XE1 = np.zeros((N, 96), dtype=np.float32)

        log("  train unit-seq GRU (4 epochs)")
        try:
            XE2 = train_gru_and_encode(unit_seqs, y, tr, u_vocab_size,
                                       maxlen=16, epochs=4, hidden=64)
            log(f"  E2 emb {XE2.shape}")
        except Exception as e:
            log(f"  E2 FAILED: {e}; falling back to zero-vector 64-d")
            XE2 = np.zeros((N, 64), dtype=np.float32)

        # Concatenate all v3 blocks (no F, no G+I)
        X_full = np.hstack([XA, XB, XC, XD_o, XCARE, XH, XE1, XE2]).astype(np.float32)
        X_no_E1  = np.hstack([XA, XB, XC, XD_o, XCARE, XH, XE2]).astype(np.float32)
        X_no_E2  = np.hstack([XA, XB, XC, XD_o, XCARE, XH, XE1]).astype(np.float32)
        X_no_E12 = np.hstack([XA, XB, XC, XD_o, XCARE, XH]).astype(np.float32)

        if fi == 0:
            log(f"  X_full shape {X_full.shape}  "
                f"E1_v3 dim {XE1.shape[1]}  E2 dim {XE2.shape[1]}")

        for name, mk in [("rf_base", lambda: learner_rf()),
                         ("rf_big",  lambda: learner_rf(big=True)),
                         ("hgb",     lambda: learner_hgb())]:
            try:
                est = CalibratedClassifierCV(mk(), method="isotonic",
                                             cv=3).fit(X_full[tr], y[tr])
                p_oof[name][te] = est.predict_proba(X_full[te])[:, 1]
                log(f"  fold {fi + 1} {name} done")
            except Exception as e:
                log(f"  fold {fi + 1} {name} FAILED: {e}")

        for Xab, arr in [(X_no_E1,  p_ab_no_E1),
                         (X_no_E2,  p_ab_no_E2),
                         (X_no_E12, p_ab_no_E12)]:
            try:
                est = CalibratedClassifierCV(learner_rf(), method="isotonic",
                                             cv=3).fit(Xab[tr], y[tr])
                arr[te] = est.predict_proba(Xab[te])[:, 1]
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
        r = eval_pooled_oof(y, p_ens, "ensemble_v3")
        log(f"  ensemble_v3 AUROC {r['auroc']:.3f} "
            f"({r['auroc_lo']:.3f}-{r['auroc_hi']:.3f}) "
            f"AUPRC {r['auprc']:.3f} F1 {r['f1']:.3f}")
        results.append(r)

    log("=== ABLATION (rf_base only) ===")
    ablation = []
    for tag, p in [("rf_base_full",     p_oof["rf_base"]),
                   ("rf_base_no_E1v3",  p_ab_no_E1),
                   ("rf_base_no_E2",    p_ab_no_E2),
                   ("rf_base_no_graphs", p_ab_no_E12)]:
        if np.isnan(p).any():
            log(f"  {tag} has NaN skipping"); continue
        au = roc_auc_score(y, p); ap = average_precision_score(y, p)
        ablation.append(dict(config=tag, auroc=au, auprc=ap))
        log(f"  {tag:22s} AUROC {au:.3f}  AUPRC {ap:.3f}")

    pd.DataFrame(results).to_csv(OUT_RES, index=False)
    pd.DataFrame(ablation).to_csv(OUT_ABL, index=False)
    np.savez(OUT_OOF, y=y,
             p_ab_no_E1=p_ab_no_E1,
             p_ab_no_E2=p_ab_no_E2,
             p_ab_no_E12=p_ab_no_E12,
             **p_oof)
    log(f"saved {OUT_RES}, {OUT_ABL}, {OUT_OOF}")
    log("=== DONE ===")


if __name__ == "__main__":
    main()
