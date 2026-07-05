"""Final push v2 — add graph blocks that were skipped in v1.

Reuses BLOCK A-D_order-D_unit_tab-F-G+I-H from analysis/final_push_080.py.

Adds:
  E1  GNN encounter embedding from A2 provider heterograph (ie-HGCN,
      per-fold trained; encounter emb + aggregated prov emb concatenated)
  E2  Care-unit-sequence GRU embedding (64-d, per-fold trained on A3
      InstitutionType tokens)

E3/E4/E5 (hypergraph, standard GNN zoo, temporal-edge) are documented as
follow-ups but not run — each requires its own graph construction and the
90-min budget forces prioritizing the two most obviously-missing blocks.
"""
from __future__ import annotations
import os, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import replace
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, f1_score, precision_score,
                             recall_score)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Reuse block builders + helpers from v1
from analysis.final_push_080 import (
    log, make_A, make_B, make_C, make_H, load_notes, make_regex_features,
    compute_bert_notes, SeqGRU, train_gru_and_encode,
    learner_rf, learner_hgb, eval_pooled_oof,
    SEED, DATA_DIR, GOLD,
)

import medhg_ps.config as C, medhg_ps.data as d
from medhg_ps.data import (fit_preprocess, apply_preprocess, load_raw,
                           build_provider_features, build_unit_features)
from medhg_ps.deploy import assemble_training_frame
from medhg_ps.graph import build_graph
from medhg_ps.train import train_model, set_seed
from medhg_ps.extract_embeddings import extract_embeddings

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
# DGL heterograph doesn't support MPS. Force CPU for the GNN pipeline.
GNN_DEV = "cpu"

OUT_LOG = Path("artifacts/newdata/final_push_080_v2.log")
OUT_RES = Path("artifacts/newdata/final_push_080_v2_results.csv")
OUT_OOF = Path("artifacts/newdata/final_push_080_v2_oof.npz")
OUT_ABL = Path("artifacts/newdata/final_push_080_v2_ablation.csv")

GNN_CFG = replace(C.DEFAULTS_TRAIN, learning_rate=0.008, max_epochs=150,
                  early_stop_patience=15, device=GNN_DEV)

VAL_FRAC = 0.10


# =====================================================================
# E1  GNN encounter embedding (per-fold trained ie-HGCN, extract emb)
# =====================================================================
def _norm(s):
    return s.astype(str).str.replace(r"\.0+$", "", regex=True)


def _agg_prov(prov_tbl, A2):
    """Mean provider embedding per encounter, joined via A2."""
    cols = [c for c in prov_tbl.columns if c.startswith("emb_")]
    p = prov_tbl.copy(); p["ProvID"] = _norm(p["ProvID"])
    a2 = A2.copy(); a2["ProvID"] = _norm(a2["ProvID"]); a2["LogID"] = _norm(a2["LogID"])
    j = a2.merge(p, on="ProvID", how="left").dropna(subset=cols, how="all")
    g = j.groupby("LogID")[cols].mean().reset_index()
    g.columns = ["LogID"] + [f"prov_emb_{i}" for i in range(len(cols))]
    return g


def train_gnn_and_encode(merged, feat_cols, cpt_arr, y, tr_idx, raw_bundle,
                         prov_ids, X_prov, unit_ids, X_unit,
                         fold_seed=SEED):
    """Train ie-HGCN on train_idx, return encounter emb + prov agg emb for all rows."""
    N = len(merged)
    rng = np.random.default_rng(fold_seed)
    tr_all = np.array(tr_idx).copy()
    rng.shuffle(tr_all)
    n_val = int(round(VAL_FRAC * N))
    val, tr = tr_all[:n_val], tr_all[n_val:]
    te = np.array([i for i in range(N) if i not in set(tr_all)], dtype=np.int64)
    tm = np.zeros(N, bool); tm[tr] = True
    vm = np.zeros(N, bool); vm[val] = True
    em = np.zeros(N, bool)
    if len(te) > 0:
        em[te] = True

    feat_all = merged[feat_cols].copy()
    _, st = fit_preprocess(feat_all.loc[tm].reset_index(drop=True), id_cols=[])
    X_enc = apply_preprocess(feat_all, st)

    art = build_graph(raw=raw_bundle, encounters_merged=merged,
                      enc_features=X_enc,
                      prov_ids=prov_ids, prov_features=X_prov,
                      unit_ids=unit_ids, unit_features=X_unit,
                      train_mask=tm, val_mask=vm, test_mask=em)
    set_seed(fold_seed)
    model, _ = train_model(art, cfg=GNN_CFG, save_dir=None, verbose=False)
    tabs = extract_embeddings(model, art, raw_prov_attrs=raw_bundle.prov_attrs,
                              raw_unit_attrs=raw_bundle.unit_attrs, device=GNN_DEV)

    # encounter embedding aligned to merged order
    enc = tabs.encounter.copy(); enc["LogID"] = _norm(enc["LogID"])
    ecols = [c for c in enc.columns if c.startswith("emb_")]
    E_enc = (merged[["LogID"]].astype(str)
             .merge(enc[["LogID"] + ecols], on="LogID", how="left"))
    E_enc_arr = E_enc[ecols].fillna(0).values.astype(np.float32)

    # aggregated provider embedding (join to A2 by ProvID -> LogID)
    prov_agg = _agg_prov(tabs.provider, raw_bundle.enc_prov_edges)
    E_prov = (merged[["LogID"]].astype(str)
              .merge(prov_agg, on="LogID", how="left"))
    prov_cols = [c for c in E_prov.columns if c.startswith("prov_emb_")]
    E_prov_arr = E_prov[prov_cols].fillna(0).values.astype(np.float32)

    return np.hstack([E_enc_arr, E_prov_arr]).astype(np.float32)


# =====================================================================
# E2  Care-unit-sequence GRU embedding (train per fold on A3 tokens)
# =====================================================================
def build_unit_seqs(merged, a3):
    """A3 InstitutionType -> tokenized per-encounter sequence.
    Uses 8 grouped tokens; median seq length ~2."""
    a3 = a3.copy()
    a3["LogID"] = a3["LogID"].astype(str)
    a3["InTime"] = pd.to_datetime(a3["InTime"], errors="coerce")
    # tokenize InstitutionType (fallback UnitType)
    tok_series = a3["InstitutionType"].fillna(a3["UnitType"]).fillna("UNK")
    vocab_map = {t: i + 1 for i, t in enumerate(sorted(tok_series.unique()))}
    a3["tid"] = tok_series.map(vocab_map).astype(np.int64)
    seqs_by = (a3.sort_values(["LogID", "InTime"])
                .groupby("LogID")["tid"].apply(list))
    all_logids = merged["LogID"].astype(str).values
    seqs = [np.asarray(seqs_by.get(lid, []), dtype=np.int64)
            for lid in all_logids]
    vocab_size = len(vocab_map) + 2
    lens = [len(s) for s in seqs]
    log(f"  unit-seq vocab={vocab_size}  median_len={int(np.median(lens))}  "
        f"p90={int(np.percentile(lens,90))}  max={max(lens)}")
    return seqs, vocab_size


# =====================================================================
# main
# =====================================================================
def main():
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    log("=== FINAL PUSH v2 (with graph blocks) START ===")

    # cohort + gold label
    merged, feat_cols, cpt_arr, Fseq, seq_all, _ = assemble_training_frame()
    gold = pd.read_parquet(GOLD)[["LogID", "ReadmittedWithin30Days_gold"]]
    gold["LogID"] = gold["LogID"].astype(str)
    merged["LogID"] = merged["LogID"].astype(str)
    merged = merged.merge(gold, on="LogID", how="left")
    y = merged["ReadmittedWithin30Days_gold"].astype(int).values
    N = len(y)
    log(f"cohort N={N} base rate {y.mean()*100:.2f}%")

    # v1 blocks (tabular, precomputed once)
    XB, colsB = None, None
    XC, colsC = None, None
    log("BLOCK B: LACE utility"); XB, colsB = make_B(merged); log(f"  B {XB.shape}")
    log("BLOCK C: LACE + HOSPITAL scores"); XC, colsC = make_C(merged); log(f"  C {XC.shape}")
    log("BLOCK H: geocode"); XH, colsH = make_H(merged); log(f"  H {XH.shape}")

    log("BLOCK G+I: regex from notes")
    notes_df = load_notes()
    all_logids = merged["LogID"].astype(str).values
    XGI, colsGI = make_regex_features(notes_df, all_logids)
    log(f"  G+I {XGI.shape}; SDoH-any="
        f"{int((XGI[:, colsGI.index('sdoh_total')] > 0).sum())}")

    log("BLOCK F: Bio_ClinicalBERT (cached load)")
    XF_raw, has_notes = compute_bert_notes(notes_df, all_logids, batch_size=32)
    log(f"  F raw {XF_raw.shape}; encounters w/ notes {int(has_notes.sum())}")

    # order sequences (block D_order)
    log("order-seq loader")
    from medhg_ps.data import load_order_sequence, collapse_order_runs
    orders_df = collapse_order_runs(load_order_sequence())
    orders_df["LogID"] = orders_df["LogID"].astype(str)
    tokens = orders_df["OrderGroup"].astype(str).fillna("UNK")
    o_vocab = {t: i + 1 for i, t in enumerate(sorted(tokens.unique()))}
    orders_df["tid"] = tokens.map(o_vocab).astype(np.int64)
    o_seqs_by = (orders_df.sort_values(["LogID", "SeqInEncounter"])
                 .groupby("LogID")["tid"].apply(list))
    order_seqs = [np.asarray(o_seqs_by.get(lid, []), dtype=np.int64)
                  for lid in all_logids]
    o_vocab_size = len(o_vocab) + 2
    log(f"  order-seq vocab={o_vocab_size}  median_len="
        f"{int(np.median([len(s) for s in order_seqs]))}")

    # Fseq care-path tabular
    Fseq["LogID"] = Fseq["LogID"].astype(str)
    care_df = pd.DataFrame({"LogID": all_logids}).merge(Fseq, on="LogID", how="left")
    care_cols = [c for c in care_df.columns if c != "LogID"]
    XCARE = care_df[care_cols].fillna(0).values.astype(np.float32)
    log(f"  care-path tabular {XCARE.shape}")

    # ---- graph substrate for E1
    log("loading raw graph tables for E1 (GNN prov)")
    raw = load_raw()
    prov_ids, X_prov, _ = build_provider_features(raw.prov_attrs)
    unit_ids, X_unit, _ = build_unit_features(raw.unit_attrs)
    log(f"  providers {len(prov_ids)}  units {len(unit_ids)}")

    # ---- A3 unit sequence for E2
    log("A3 unit trajectory for E2")
    a3 = d._read_table(C.UNIT_EDGES_PARQUET, C.A3_UNIT_EDGES_COLUMNS)
    unit_seqs, u_vocab_size = build_unit_seqs(merged, a3)

    # ---- 5-fold CV
    log("=== CROSS VALIDATION (5-fold seed 42) ===")
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    p_oof = {"rf_base": np.full(N, np.nan),
             "rf_big":  np.full(N, np.nan),
             "hgb":     np.full(N, np.nan)}

    # ablation OOFs (drop E1 alone, drop E2 alone, drop both graphs)
    p_ab_no_E1  = np.full(N, np.nan)
    p_ab_no_E2  = np.full(N, np.nan)
    p_ab_no_E12 = np.full(N, np.nan)  # = v1 configuration

    for fi, (tr, te) in enumerate(skf.split(np.zeros(N), y)):
        log(f"--- fold {fi + 1}/5  train {len(tr)} test {len(te)} ---")
        train_mask = np.zeros(N, bool); train_mask[tr] = True

        # A base
        XA = make_A(merged, feat_cols, cpt_arr, train_mask)
        # D_order: order-GRU per fold
        log(f"  train order-GRU (epochs 5)")
        XD_o = train_gru_and_encode(order_seqs, y, tr, o_vocab_size,
                                    maxlen=128, epochs=5, hidden=64)
        # F: PCA fit on train
        pca = PCA(n_components=64, random_state=SEED).fit(XF_raw[tr])
        XF = np.hstack([pca.transform(XF_raw),
                        has_notes.reshape(-1, 1)]).astype(np.float32)

        # E1: GNN prov embedding
        log(f"  train GNN (ie-HGCN)")
        try:
            XE1 = train_gnn_and_encode(merged, feat_cols, cpt_arr, y, tr,
                                       raw, prov_ids, X_prov,
                                       unit_ids, X_unit,
                                       fold_seed=SEED + fi)
            log(f"  E1 emb {XE1.shape}")
        except Exception as e:
            log(f"  E1 FAILED: {e}; falling back to zero-vector 96-d")
            XE1 = np.zeros((N, 96), dtype=np.float32)

        # E2: unit-seq GRU
        log(f"  train unit-seq GRU (epochs 4)")
        try:
            XE2 = train_gru_and_encode(unit_seqs, y, tr, u_vocab_size,
                                       maxlen=16, epochs=4, hidden=64)
            log(f"  E2 emb {XE2.shape}")
        except Exception as e:
            log(f"  E2 FAILED: {e}; falling back to zero-vector 64-d")
            XE2 = np.zeros((N, 64), dtype=np.float32)

        # Concatenate ALL blocks (v2 = v1 + E1 + E2)
        X_full = np.hstack([XA, XB, XC, XD_o, XCARE, XF, XGI, XH,
                            XE1, XE2]).astype(np.float32)
        # Ablation variants
        X_no_E1  = np.hstack([XA, XB, XC, XD_o, XCARE, XF, XGI, XH,
                              XE2]).astype(np.float32)
        X_no_E2  = np.hstack([XA, XB, XC, XD_o, XCARE, XF, XGI, XH,
                              XE1]).astype(np.float32)
        X_no_E12 = np.hstack([XA, XB, XC, XD_o, XCARE, XF, XGI, XH]).astype(np.float32)

        if fi == 0:
            log(f"  X_full shape {X_full.shape}  "
                f"E1 dim {XE1.shape[1]}  E2 dim {XE2.shape[1]}")

        # Fit 3 learners on X_full
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

        # Ablation with rf_base only
        for Xab, arr in [(X_no_E1,  p_ab_no_E1),
                         (X_no_E2,  p_ab_no_E2),
                         (X_no_E12, p_ab_no_E12)]:
            try:
                est = CalibratedClassifierCV(learner_rf(), method="isotonic",
                                             cv=3).fit(Xab[tr], y[tr])
                arr[te] = est.predict_proba(Xab[te])[:, 1]
            except Exception as e:
                log(f"  ablation FAILED: {e}")

    # ---- eval
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

    # ensemble
    valid = [k for k in p_oof if not np.isnan(p_oof[k]).any()]
    if len(valid) >= 2:
        p_ens = np.mean([p_oof[k] for k in valid], axis=0)
        r = eval_pooled_oof(y, p_ens, "ensemble_v2")
        log(f"  ensemble_v2 AUROC {r['auroc']:.3f} "
            f"({r['auroc_lo']:.3f}-{r['auroc_hi']:.3f}) "
            f"AUPRC {r['auprc']:.3f} F1 {r['f1']:.3f}")
        results.append(r)

    # Ablation results
    log("=== ABLATION (rf_base only) ===")
    ablation = []
    for tag, p in [("rf_base_full",   p_oof["rf_base"]),
                   ("rf_base_no_E1",  p_ab_no_E1),
                   ("rf_base_no_E2",  p_ab_no_E2),
                   ("rf_base_no_E12", p_ab_no_E12)]:
        if np.isnan(p).any():
            log(f"  {tag} has NaN skipping"); continue
        au = roc_auc_score(y, p); ap = average_precision_score(y, p)
        ablation.append(dict(config=tag, auroc=au, auprc=ap))
        log(f"  {tag:20s} AUROC {au:.3f}  AUPRC {ap:.3f}")

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
