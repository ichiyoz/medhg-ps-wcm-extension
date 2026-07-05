"""Phenotypes v15 — DICE on TWO substrates for the unified-hospital-operations
concept, side by side:

  Substrate A: multi-attribute flow GRU (v9b-style) — chronological order-event
               sequence with time buckets and order source, trained supervised
               on readmission. Final GRU hidden state = 64-d embedding.

  Substrate B: MedHG-PS ie-HGCN encoder on the encounter + provider + unit +
               order-group heterograph. Trained supervised on readmission.
               Encoder encounter embedding = 64-d.

Both fed to the SAME DICE config that produced clean separation in prior
3-substrate work:
  K = 3, d = 16, lam_smd = 30, smd_target = 0.30, spread = 0.0

Cohort: N=13,858 (Expired/Hospice/Acute/AMA excluded).
Reports per-phenotype size, readmission rate, and clinical descriptors
(median LOS, age, order count, provider role mix, top orders, first unit).
"""
from __future__ import annotations
import sys, time, warnings
from copy import copy as shallow_copy
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.final_push_080 import log, SEED, DATA_DIR, GOLD
from analysis.final_push_080_v8 import _norm_id, prov_role, PROV_ROLES
from analysis.final_push_080_v9b import (
    build_sequences, train_gru_and_encode, GRU_HIDDEN,
)
from analysis.final_push_080_v2 import train_gnn_and_encode
from analysis.final_push_080_v3 import build_order_group_substrate
from analysis import dice as dice_mod
from analysis.dice import fit as dice_fit, cluster_proba

from medhg_ps.data import (load_raw, load_order_sequence, collapse_order_runs,
                            build_provider_features)
from medhg_ps.deploy import assemble_training_frame

OUT_LOG = Path("artifacts/newdata/phenotypes_v15.log")
OUT_DIR = Path("artifacts/newdata/phenotypes_v15")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE_LABELS = {"Expired","Expired in Medical Facility","Hospice/Home",
                  "Hospice/Medical Facility","Acute / Short Term Hospital",
                  "Left Against Medical Advice"}


def describe(hard, y, merged, orders_df, a3, a2):
    """Compact phenotype description used for BOTH substrates."""
    K = int(hard.max()) + 1
    N = len(y)
    lid2idx = {lid: i for i, lid in enumerate(merged["LogID"].astype(str).values)}

    # LOS from A3
    a3l = a3.copy()
    a3l["InTime"] = pd.to_datetime(a3l["InTime"], errors="coerce")
    a3l["OutTime"] = pd.to_datetime(a3l["OutTime"], errors="coerce")
    los = a3l.groupby("LogID").agg(
        f=("InTime","min"), l=("OutTime","max")).reset_index()
    los["los_hr"] = (los["l"] - los["f"]).dt.total_seconds() / 3600
    los["enc_i"] = los["LogID"].astype(str).map(lid2idx)
    los_arr = np.zeros(N, dtype=np.float32)
    ok = los.dropna(subset=["enc_i"])
    los_arr[ok["enc_i"].astype(int).values] = ok["los_hr"].values.astype(np.float32)

    # First unit
    a3l = a3l.dropna(subset=["InTime"]).sort_values(["LogID","InTime"])
    fu = a3l.groupby("LogID").first()["DepartmentID"].astype(str)
    first_unit = pd.DataFrame({"LogID": fu.index, "u": fu.values})
    first_unit["enc_i"] = first_unit["LogID"].astype(str).map(lid2idx)
    fu_arr = np.array(["UNK"]*N, dtype=object)
    ok = first_unit.dropna(subset=["enc_i"])
    fu_arr[ok["enc_i"].astype(int).values] = ok["u"].values

    # Orders per encounter + top OG per encounter
    orders_df["enc_i"] = orders_df["LogID"].astype(str).map(lid2idx)
    n_ord = orders_df.groupby("LogID").size().reset_index(name="n")
    n_ord["enc_i"] = n_ord["LogID"].astype(str).map(lid2idx)
    n_ord_arr = np.zeros(N, dtype=np.float32)
    ok = n_ord.dropna(subset=["enc_i"])
    n_ord_arr[ok["enc_i"].astype(int).values] = ok["n"].values.astype(np.float32)

    # Provider role counts per encounter
    role_arr = {r: np.zeros(N, dtype=np.float32) for r in PROV_ROLES}
    a2 = a2.copy()
    a2["enc_i"] = a2["LogID"].astype(str).map(lid2idx)
    for r in PROV_ROLES:
        sub = a2[a2["role"] == r].dropna(subset=["enc_i"])
        cnt = sub.groupby("enc_i").size()
        role_arr[r][cnt.index.astype(int).values] = cnt.values.astype(np.float32)

    age = pd.to_numeric(merged["AgeYears"], errors="coerce").fillna(60).values
    sex_f = (merged["Gender"].astype(str) == "F").astype(float).values
    home = merged["Discharge Disposition"].astype(str).isin(
        {"Home or Self Care","Home-Health Care Svc"}).values.astype(float)

    rows = []
    for k in range(K):
        m = hard == k
        n = int(m.sum())
        if n == 0: continue
        rate = float(y[m].mean())
        # Top-5 orders
        top5 = (orders_df[orders_df["enc_i"].isin(np.where(m)[0])]
                .groupby("OrderGroup").size().sort_values(ascending=False).head(5))
        top5_str = "; ".join(f"{k[:32]} ({v:,})" for k, v in top5.items())
        # First unit mode
        fu_mode = pd.Series(fu_arr[m]).value_counts().head(1).index[0] if m.any() else "UNK"
        rows.append(dict(
            phenotype=f"cluster_{k}",
            n=n, readmit_rate=rate,
            median_los_hr=float(np.nanmedian(los_arr[m])) if m.any() else np.nan,
            median_age=float(np.median(age[m])),
            fraction_female=float(sex_f[m].mean()),
            fraction_home=float(home[m].mean()),
            median_n_orders=float(np.median(n_ord_arr[m])),
            median_n_attending=float(np.median(role_arr["attending"][m])),
            median_n_resident=float(np.median(role_arr["resident"][m])),
            median_n_crna=float(np.median(role_arr["crna"][m])),
            median_n_anesth=float(np.median(role_arr["anesth"][m])),
            most_common_first_unit=str(fu_mode),
            top5_order_groups=top5_str,
        ))
    df = pd.DataFrame(rows).sort_values("readmit_rate").reset_index(drop=True)
    df["phenotype"] = [f"P{i+1} ({name})" for i, name in
                       zip(range(len(df)), ["low","moderate","high"][:len(df)])]
    return df


def run_dice(emb, y, K=3, d=16):
    scaler = StandardScaler(); emb_std = scaler.fit_transform(emb).astype(np.float32)
    dice_mod.LAM["smd"] = 30.0
    t0 = time.time()
    model = dice_fit(X=emb_std, y=y.astype(np.float32), K=K, d=d,
                     smd_target=0.30, spread=0.0, verbose=False, seed=SEED)
    P = cluster_proba(model, emb_std)
    hard = P.argmax(axis=1)
    log(f"  DICE fit in {time.time()-t0:.0f}s  sizes={np.bincount(hard, minlength=K)}")
    for k in range(K):
        m = hard == k
        log(f"    cluster {k}: n={int(m.sum()):,}  readmit={y[m].mean()*100:.2f}%")
    return hard


def main():
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    log("=== PHENOTYPES v15 (flow-GRU + ie-HGCN substrates) START ===")

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

    # Ancillary tables
    orders_df = collapse_order_runs(load_order_sequence())
    raw = load_raw()
    a3 = pd.read_parquet(f"{DATA_DIR}/A3_enc_unit_edges.parquet")
    a3["LogID"] = a3["LogID"].astype(str)
    # Filter raw tables to kept cohort
    kept = set(_norm_id(merged["LogID"]).values)
    ep = raw.enc_prov_edges.copy(); ep["LogID"] = _norm_id(ep["LogID"])
    ep = ep[ep["LogID"].isin(kept)]
    raw_f = shallow_copy(raw)
    raw_f.enc_prov_edges = ep

    # For descriptions: provider role table
    a4 = raw.prov_attrs.copy(); a4["ProvID"] = _norm_id(a4["ProvID"])
    a4["role"] = a4["ProvType"].apply(prov_role)
    a2_desc = ep.copy()
    a2_desc["ProvID"] = _norm_id(a2_desc["ProvID"])
    a2_desc = a2_desc.merge(a4[["ProvID","role"]], on="ProvID", how="left")
    a2_desc["role"] = a2_desc["role"].fillna("other")

    # ====================================================================
    # SUBSTRATE A: multi-attribute flow GRU on order sequence
    # ====================================================================
    log("=== SUBSTRATE A: multi-attribute flow GRU on order sequence ===")
    all_logids = _norm_id(merged["LogID"]).values
    lid2idx = {lid: i for i, lid in enumerate(all_logids)}
    order_seqs, time_seqs, src_seqs, lens, n_ord_vocab, n_src_vocab = \
        build_sequences(orders_df, lid2idx, N)
    log(f"  sequences built  median len {int(np.median(lens))}  "
        f"order vocab {n_ord_vocab}  source vocab {n_src_vocab}")
    # Train the GRU on ALL rows (descriptive phenotype substrate — no CV)
    tr_all = np.arange(N)
    emb_A = train_gru_and_encode(
        order_seqs, time_seqs, src_seqs, lens, y, tr_all,
        n_ord_vocab, n_src_vocab, epochs=6, seed=SEED)
    log(f"  A embedding {emb_A.shape}")

    log("  running DICE on Substrate A")
    hard_A = run_dice(emb_A, y)
    desc_A = describe(hard_A, y, merged, orders_df.copy(), a3, a2_desc)
    outA = OUT_DIR / "phenotypes_substrate_A_flow_gru.csv"
    desc_A.to_csv(outA, index=False)
    log(f"  saved {outA}")
    print(desc_A.to_string(index=False))

    # ====================================================================
    # SUBSTRATE B: MedHG-PS ie-HGCN encoder on unified heterograph
    # ====================================================================
    log("\n=== SUBSTRATE B: MedHG-PS ie-HGCN on encounter+provider+order-group ===")
    # Reuse v3's order-group substrate to reduce risk of new bugs.
    # ie-HGCN encoder with orders as A3 was the closest match to a "unified"
    # heterograph the paper's model supports directly.
    prov_ids, X_prov, _ = build_provider_features(raw_f.prov_attrs)
    raw_og, og_ids, X_og = build_order_group_substrate(raw_f, merged, orders_df)
    tr_all = np.arange(N)
    emb_B = train_gnn_and_encode(
        merged, feat_cols, cpt_arr, y, tr_all, raw_og,
        prov_ids, X_prov, og_ids, X_og, fold_seed=SEED)
    log(f"  B embedding {emb_B.shape}")

    log("  running DICE on Substrate B")
    hard_B = run_dice(emb_B, y)
    desc_B = describe(hard_B, y, merged, orders_df.copy(), a3, a2_desc)
    outB = OUT_DIR / "phenotypes_substrate_B_iehgcn.csv"
    desc_B.to_csv(outB, index=False)
    log(f"  saved {outB}")
    print(desc_B.to_string(index=False))

    # Save embeddings and cluster assignments
    np.savez(OUT_DIR / "embeddings_and_clusters.npz",
             y=y, emb_A=emb_A, emb_B=emb_B, hard_A=hard_A, hard_B=hard_B)
    log("=== DONE ===")


if __name__ == "__main__":
    main()
