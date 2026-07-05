"""PLOS reproduction of Chen et al. ie-HGCN paper setup.

Reproduces the paper's design faithfully:
  * Outcome: PLOS = (LOS > 75th pct) — top-quartile of total encounter LOS.
  * Encounter node features include the paper's `feats_icu` summary: per-encounter
    aggregation of A3 unit-stays into acute_count/hours, intensive_count/hours,
    intermediate_count/hours, plus total_count/hours. These are WHOLE-ENCOUNTER
    aggregates -- they include post-op unit stays and therefore MECHANICALLY
    encode LOS. Included here to faithfully mirror the paper; leakage is noted.
  * Provider-Provider Graph (PPG) + ICU-graph (ICUG) heterograph via the existing
    MedHG-PS builder (encounter/provider/unit tri-partite).
  * ie-HGCN trained end-to-end, per-fold isotonic-calibrated pooled OOF,
    bootstrap n=2000 CIs.

Rows reported:
  1. medhgps_A3summary_PPG_PLOS  — ie-HGCN + tabular + A3 summary + PPG+ICUG
  2. rf_clin_leaky               — RF on tabular + A3 summary (same features, no graph)

Prior baseline (no A3 summary, truly pre-op): rf_clin AUROC 0.962 / AUPRC 0.881 / F1 0.812
Prior collapse (A2-only PPG, no A3 features): medhgps_raw AUROC 0.500
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, f1_score, confusion_matrix)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

import medhg_ps.config as C
import medhg_ps.data as D
from medhg_ps.data import (fit_preprocess, apply_preprocess,
                           build_provider_features, build_unit_features,
                           load_raw)
from medhg_ps.deploy import assemble_training_frame
from medhg_ps.evaluate import _bootstrap_ci
from medhg_ps.graph import build_graph
from medhg_ps.train import train_model, set_seed

OUT_DIR = Path("artifacts/newdata"); OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG = OUT_DIR / "plos_medhgps_paper.log"
def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f: f.write(msg + "\n")
open(LOG, "w").close()

SEED = 42
K = 5
VAL_FRAC = 0.10
LEAKY = {"Discharge Disposition",
         "# of Cardiac Arrest Requiring CPR",
         "# of Stroke/Cerebral Vascular Acccident (CVA)",
         "# of Postop Unplanned Intubation",
         "preop_los_acute_hr","preop_los_intensive_hr","preop_los_intermediate_hr",
         "preop_transfer_count","preop_n_units"}

# ================= data + PLOS + A3 summary ==============================
log("[load] cohort + PLOS label + A3 summary")
merged, feat_cols, cpt_arr, Fseq, seq_all, _ = assemble_training_frame()
merged["LogID"] = merged["LogID"].astype(str)
merged["_row"] = np.arange(len(merged))

a3 = D._read_table(C.UNIT_EDGES_PARQUET, C.A3_UNIT_EDGES_COLUMNS).copy()
a3["LogID"]   = a3["LogID"].astype(str)
a3["InTime"]  = pd.to_datetime(a3["InTime"], errors="coerce")
a3["OutTime"] = pd.to_datetime(a3["OutTime"], errors="coerce")
a3["Hours"]   = pd.to_numeric(a3["Hours"], errors="coerce").fillna(0.0)
los_days = a3.groupby("LogID").apply(
    lambda g: (g["OutTime"].max() - g["InTime"].min()).total_seconds() / 86400
).rename("los_days").reset_index()

def _summary(g):
    acute = g.loc[g["UnitType"] == "Acute", "Hours"]
    inten = g.loc[g["UnitType"] == "Intensive", "Hours"]
    other = g.loc[g["UnitType"] == "Other", "Hours"]  # paper's "intermediate"
    return pd.Series(dict(
        acute_count=int(len(acute)),   acute_hours=float(acute.sum()),
        intensive_count=int(len(inten)), intensive_hours=float(inten.sum()),
        intermediate_count=int(len(other)), intermediate_hours=float(other.sum()),
        total_count=int(len(g)),        total_hours=float(g["Hours"].sum()),
    ))
feats_a3 = a3.groupby("LogID").apply(_summary).reset_index()

log(f"[audit] acute_hours: median {feats_a3['acute_hours'].median():.1f}  p90 {feats_a3['acute_hours'].quantile(0.9):.1f}  max {feats_a3['acute_hours'].max():.0f}")
log(f"[audit] intensive_hours: median {feats_a3['intensive_hours'].median():.1f}  p90 {feats_a3['intensive_hours'].quantile(0.9):.1f}  max {feats_a3['intensive_hours'].max():.0f}")
log(f"[audit] intermediate_hours (Other): median {feats_a3['intermediate_hours'].median():.1f}  p90 {feats_a3['intermediate_hours'].quantile(0.9):.1f}  max {feats_a3['intermediate_hours'].max():.0f}")
log(f"[audit] total_hours: median {feats_a3['total_hours'].median():.1f}  p90 {feats_a3['total_hours'].quantile(0.9):.1f}  max {feats_a3['total_hours'].max():.0f}")

merged = merged.merge(los_days, on="LogID", how="left")
merged = merged.merge(feats_a3, on="LogID", how="left")
mask = merged["los_days"].notna()
merged = merged.loc[mask].reset_index(drop=True)
cpt_arr = cpt_arr[merged["_row"].values]

cutoff = merged["los_days"].quantile(0.75)
y = (merged["los_days"] > cutoff).astype(int).values
N = len(y)
log(f"[cohort] N={N:,}  PLOS cutoff={cutoff:.2f} days  event={y.mean()*100:.2f}%")

A3_SUMMARY_COLS = ["acute_count","acute_hours","intensive_count","intensive_hours",
                   "intermediate_count","intermediate_hours","total_count","total_hours"]
for c_ in A3_SUMMARY_COLS: merged[c_] = merged[c_].fillna(0.0)

clean_feats = [f for f in feat_cols if f not in LEAKY]
enc_feat_cols = clean_feats + A3_SUMMARY_COLS
log(f"[features] tabular clean={len(clean_feats)}  A3 summary={len(A3_SUMMARY_COLS)}  CPT one-hot n={len(np.unique(cpt_arr))}")

# ================= metric helpers ========================================
def metrics_at_max_f1(y_true, p):
    thr_grid = np.unique(np.concatenate([np.linspace(0.05, 0.95, 91),
                                         np.quantile(p, np.linspace(0.01, 0.99, 99))]))
    best_f1, best_thr = -1, 0.5
    for t in thr_grid:
        pred = (p >= t).astype(int)
        if pred.sum() == 0 or pred.sum() == len(pred): continue
        f = f1_score(y_true, pred)
        if f > best_f1: best_f1, best_thr = f, t
    pred = (p >= best_thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return dict(thr=best_thr, f1=best_f1,
                precision=tp/max(tp+fp,1), recall=tp/max(tp+fn,1),
                specificity=tn/max(tn+fp,1), flag_rate=pred.mean())

def eval_row(name, p):
    au = roc_auc_score(y, p); ap = average_precision_score(y, p); br = brier_score_loss(y, p)
    au_ci = _bootstrap_ci(y, p, roc_auc_score, n_boot=2000, seed=0)
    ap_ci = _bootstrap_ci(y, p, average_precision_score, n_boot=2000, seed=1)
    m = metrics_at_max_f1(y, p)
    thr = m["thr"]
    def f1_at(y_, p_): return f1_score(y_, (p_ >= thr).astype(int))
    f1_ci = _bootstrap_ci(y, p, f1_at, n_boot=2000, seed=2)
    row = dict(model=name,
               AUROC=au, AUROC_ci_lo=au_ci[0], AUROC_ci_hi=au_ci[1],
               AUPRC=ap, AUPRC_ci_lo=ap_ci[0], AUPRC_ci_hi=ap_ci[1],
               Brier=br, thr=thr,
               F1=m["f1"], F1_ci_lo=f1_ci[0], F1_ci_hi=f1_ci[1],
               precision=m["precision"], recall=m["recall"],
               specificity=m["specificity"], flag_rate=m["flag_rate"])
    log(f"[metrics] {name:34s} AUROC {au:.3f} ({au_ci[0]:.3f}-{au_ci[1]:.3f})  "
        f"AUPRC {ap:.3f} ({ap_ci[0]:.3f}-{ap_ci[1]:.3f})  Brier {br:.4f}  "
        f"F1 {m['f1']:.3f} ({f1_ci[0]:.3f}-{f1_ci[1]:.3f}) @ thr {thr:.3f}  "
        f"P {m['precision']:.3f} R {m['recall']:.3f} S {m['specificity']:.3f} flag {m['flag_rate']*100:.1f}%")
    return row

def build_enc_X(train_idx):
    df = merged[enc_feat_cols]
    _, st = fit_preprocess(df.iloc[train_idx].reset_index(drop=True), id_cols=[])
    Xa = apply_preprocess(df, st)
    oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(cpt_arr[train_idx])
    Xc = oh.transform(cpt_arr)
    return np.hstack([Xa, Xc])

# ================= Row 2 (quick): rf_clin_leaky ==========================
log("\n=== rf_clin_leaky: RF on [tabular + A3 summary + CPT] ===")
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
p_rf = np.full(N, np.nan)
for fold, (tr, te) in enumerate(skf.split(np.zeros(N), y), 1):
    X = build_enc_X(tr)
    rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=10, max_features="sqrt",
                                class_weight="balanced", random_state=42, n_jobs=-1)
    est = CalibratedClassifierCV(rf, method="isotonic", cv=3).fit(X[tr], y[tr])
    p_rf[te] = est.predict_proba(X[te])[:, 1]
    log(f"[fold {fold}] rf test AUROC {roc_auc_score(y[te], p_rf[te]):.3f}")
row_rf = eval_row("rf_clin_leaky", p_rf)

# ================= Row 1: paper-style ie-HGCN ============================
log("\n=== ie-HGCN (paper-style): PPG + ICU-graph, A3 summary on encounter nodes ===")
# DGL doesn't support MPS; fall back to CUDA if available else CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["MEDHG_PS_DEVICE"] = device
log(f"[device] {device} (DGL has no MPS backend)")

raw = load_raw()
# Restrict raw to matched LogIDs
kept = set(merged["LogID"].tolist())
raw.encounters      = raw.encounters[raw.encounters["LogID"].astype(str).isin(kept)].reset_index(drop=True)
raw.enc_prov_edges  = raw.enc_prov_edges[raw.enc_prov_edges["LogID"].astype(str).isin(kept)].reset_index(drop=True)
raw.enc_unit_edges  = raw.enc_unit_edges[raw.enc_unit_edges["LogID"].astype(str).isin(kept)].reset_index(drop=True)

prov_ids, X_prov, _ = build_provider_features(raw.prov_attrs)
unit_ids, X_unit, _ = build_unit_features(raw.unit_attrs)
log(f"[graph] enc={N} prov={len(prov_ids)} unit={len(unit_ids)}  prov_dim={X_prov.shape[1]}  unit_dim={X_unit.shape[1]}")

# Reorder merged so its row order matches raw.encounters
enc_order = raw.encounters[["LogID"]].copy()
enc_order["LogID"] = enc_order["LogID"].astype(str)
idx_map = {lid: i for i, lid in enumerate(merged["LogID"].tolist())}
order_idx = enc_order["LogID"].map(idx_map).astype(int).values
merged_ord = merged.iloc[order_idx].reset_index(drop=True).copy()
cpt_arr_ord = cpt_arr[order_idx]
y_ord = y[order_idx]
# add label column expected by build_graph
merged_ord["ReadmittedWithin30Days"] = y_ord

# Configure ie-HGCN training
cfg = C.TrainConfig(
    learning_rate=1e-3, l2_reg=1e-4, dropout=0.3, batch_norm=True,
    hidden_dim_1=128, hidden_dim_2=64, hidden_dim_3=32, attn_dim=32, n_layers=3,
    max_epochs=80, early_stop_patience=15, resampling="none",
    train_frac=1-VAL_FRAC-1/K, val_frac=VAL_FRAC, test_frac=1/K,
    split_seed=SEED, device=device,
)

p_gnn_ord = np.full(N, np.nan)
skf2 = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
for fold, (tr_all_ord, te_ord) in enumerate(skf2.split(np.zeros(N), y_ord), 1):
    rng = np.random.default_rng(SEED + fold)
    tr_all_ord = tr_all_ord.copy(); rng.shuffle(tr_all_ord)
    n_val = int(round(VAL_FRAC * N))
    va_ord = tr_all_ord[:n_val]
    tr_ord = tr_all_ord[n_val:]

    train_mask = np.zeros(N, dtype=bool); train_mask[tr_ord] = True
    val_mask   = np.zeros(N, dtype=bool); val_mask[va_ord]   = True
    test_mask  = np.zeros(N, dtype=bool); test_mask[te_ord]  = True

    # Build enc features on this fold's train set (in ordered index space)
    # Convert ordered indices back to merged (unordered) indices for preprocessing helper
    tr_unord = order_idx[tr_ord]
    X_enc_ord_all = build_enc_X(tr_unord)[order_idx]

    artifacts = build_graph(raw=raw, encounters_merged=merged_ord, enc_features=X_enc_ord_all,
                            prov_ids=prov_ids, prov_features=X_prov,
                            unit_ids=unit_ids, unit_features=X_unit,
                            train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    set_seed(SEED + fold)
    model, trr = train_model(artifacts, cfg=cfg, save_dir=None, verbose=False)

    # Predict on test
    model.eval()
    g_dev = artifacts.g.to(device)
    with torch.no_grad():
        logits, _ = model.to(device)(g_dev, {nt: g_dev.nodes[nt].data["h"] for nt in C.NODE_TYPES})
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    # Isotonic calibration using val slice
    try:
        ir = IsotonicRegression(out_of_bounds="clip").fit(probs[val_mask], y_ord[val_mask])
        p_calib = ir.transform(probs)
    except Exception:
        p_calib = probs
    p_gnn_ord[te_ord] = p_calib[te_ord]
    log(f"[fold {fold}] ie-HGCN test AUROC {roc_auc_score(y_ord[te_ord], p_gnn_ord[te_ord]):.3f}  "
        f"(best_val AUROC {trr.best_val_auroc:.3f} @ epoch {trr.best_epoch})")

# Map back from ordered indices to merged (unordered) indices
inv = np.empty(N, dtype=int); inv[order_idx] = np.arange(N)
p_gnn = p_gnn_ord[inv]
row_gnn = eval_row("medhgps_A3summary_PPG_PLOS", p_gnn)

# ================= save ==================================================
results = pd.DataFrame([row_gnn, row_rf])
csv_path = OUT_DIR / "plos_medhgps_paper_results.csv"
results.to_csv(csv_path, index=False)
np.savez(OUT_DIR / "plos_medhgps_paper_oof.npz",
         y=y, rf_clin_leaky=p_rf, medhgps_A3summary_PPG_PLOS=p_gnn)
log(f"\n[save] {csv_path}")

# ================= summary ===============================================
log(f"\n=== SUMMARY vs prior rf_clin (no A3 summary): AUROC 0.962 / AUPRC 0.881 / F1 0.812 ===")
log(f"  rf_clin_leaky  (RF + A3 summary):    AUROC {row_rf['AUROC']:.3f}  AUPRC {row_rf['AUPRC']:.3f}  F1 {row_rf['F1']:.3f}")
log(f"  medhgps_paper  (ie-HGCN + A3 sum):   AUROC {row_gnn['AUROC']:.3f}  AUPRC {row_gnn['AUPRC']:.3f}  F1 {row_gnn['F1']:.3f}")
log(f"  Δ(rf_leaky vs rf_clin no-A3): AUROC {row_rf['AUROC']-0.962:+.3f}  AUPRC {row_rf['AUPRC']-0.881:+.3f}")
log(f"  Δ(gnn vs rf_leaky):           AUROC {row_gnn['AUROC']-row_rf['AUROC']:+.3f}  AUPRC {row_gnn['AUPRC']-row_rf['AUPRC']:+.3f}")
log("[done]")
