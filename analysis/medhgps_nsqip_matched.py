"""MedHG-PS on the 1,583 WCM<->NSQIP matched cohort.

Tests whether the MedHG-PS heterograph approach benefits from gold-standard
NSQIP-abstracted features vs WCM-Clarity-derived ones, on the same patients
with the same (NSQIP) outcome.

Models (all 5-fold seed 42, isotonic-calibrated pooled OOF, bootstrap n=2000):
  A. RF on NSQIP features (baseline)
  B. RF on WCM features (feature-quality contrast)
  C. ie-HGCN raw with NSQIP encounter node features
  D. ie-HGCN raw with WCM encounter node features
  E. NSQIP RF stacked with ie-HGCN encounter embedding
"""
from __future__ import annotations
import re, numpy as np, pandas as pd, torch
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import OneHotEncoder

import medhg_ps.config as C
from medhg_ps.data import (load_raw, fit_preprocess, apply_preprocess,
                           build_provider_features, build_unit_features,
                           build_preop_trajectory_features, add_calendar_features)
from medhg_ps.graph import build_graph
from medhg_ps.train import train_model, set_seed, _resolve_device
from medhg_ps.extract_embeddings import extract_embeddings
from medhg_ps.evaluate import _bootstrap_ci

SEED = 42; K = 5; VAL_FRAC = 0.10
EXT = "/Users/yiyezhang/Downloads/Case_Details_and_Custom_Fields_Report-01-Apr-2025-0916.xlsx"
GOLD_PARQUET = "/Users/yiyezhang/Downloads/medhg_ps_data/bulk_features_with_label_gold.parquet"

# ------------------------------------------------------------
# 1. Load + match on (MRN, Operation Date)
# ------------------------------------------------------------
def _nm(s): return s.astype(str).str.strip().str.replace(r"\.0+$","",regex=True).str.lstrip("0").replace({"":np.nan,"nan":np.nan})
def _disp(s):
    s = str(s).lower()
    if "ama" in s or "against medical" in s: return "AMA"
    if "expired" in s: return "Expired"
    if "home" in s or "self care" in s or "hospice/home" in s: return "Home"
    if any(k in s for k in ["facility","hospital","nursing","rehab","custodial","skilled"]): return "Facility"
    return "Other"

print("[t] loading WCM bulk (gold) + NSQIP xlsx + graph tables...", flush=True)
bulk = pd.read_parquet(GOLD_PARQUET)
bulk["LogID"] = bulk["LogID"].astype(str)
bulk["mrn"] = _nm(bulk["PAT_MRN_ID"])
bulk["dt"]  = pd.to_datetime(bulk["SurgeryDate"], errors="coerce").dt.date

ext = pd.read_excel(EXT)
wunit = ext.get("Weight Unit", pd.Series(["kg"]*len(ext))).astype(str).str.lower()
ext["mrn"] = _nm(ext["MRN"])
ext["dt"]  = pd.to_datetime(ext["Operation Date"], errors="coerce").dt.date
ext = ext.rename(columns=C.NSQIP_TO_SQL_RENAMES)
ext["Gender"] = ext["Gender"].map({"Female":"F","Male":"M"}).fillna(ext["Gender"])
ext["PatientType"] = ext["PatientType"].map({"Inpatient":"I","Outpatient":"O"}).fillna(ext["PatientType"])
_roman = {"I":"1.0","II":"2.0","III":"3.0","IV":"4.0","V":"5.0","VI":"6.0"}
ext["ASAClass"] = ext["ASAClass"].map(lambda s:(lambda m:_roman.get(m.group(1)) if m else np.nan)(re.search(r"ASA\s+(VI|IV|V|III|II|I)\b",str(s))))
_anes = {"General":"general","Regional":"regional","Spinal":"spinal","Epidural":"epidural","Monitored anesthesia care/IV sedation":"MAC"}
ext["AnesType"] = ext["AnesType"].map(_anes).fillna(ext["AnesType"].astype(str).str.lower())
ext["Discharge Disposition"] = ext["Discharge Disposition"].map(_disp)
w = pd.to_numeric(ext["Weight (kg)"], errors="coerce")
ext["Weight (kg)"] = np.where(wunit.str.startswith("lb"), w * 0.453592, w)
ext["PrimaryCPT"] = ext.get("CPT Code", pd.Series([np.nan]*len(ext))).astype(str).str.replace(r"\.0+$","",regex=True)
ext["y_nsqip"] = (pd.to_numeric(ext.get("# of Unplanned Readmissions", 0), errors="coerce").fillna(0) > 0).astype(int)

# Also coarse-bucket WCM disp so features compare on same scale
bulk["Discharge Disposition"] = bulk["Discharge Disposition"].map(_disp)

# Common feature set (in both sources)
SDOH = {"Language","SDOH_Housing_Z","SDOH_Food_Z","SDOH_Financial_Z","SDOH_Any_Z"}
feats = [f for f in C.MODEL_FEATURE_COLUMNS
         if f not in SDOH and f in ext.columns and f in bulk.columns]
print(f"[t] common feature set (n={len(feats)}): {feats}", flush=True)

# Merge — single-clean-merge with suffixes
key = ["mrn","dt"]
b_slim = bulk[key + ["LogID"] + feats].drop_duplicates(key, keep="first")
e_slim = ext[key + feats + ["y_nsqip"]].drop_duplicates(key, keep="first")
m = b_slim.merge(e_slim, on=key, how="inner", suffixes=("__wcm","__nsqip"))
print(f"[t] matched cases: {len(m):,}", flush=True)
y = m["y_nsqip"].astype(int).values
print(f"[t] gold outcome rate: {y.mean()*100:.2f}%   positives {int(y.sum())}", flush=True)

# The two aligned tabular views (rows in same order as m)
wcm_view   = m[[c+"__wcm"   for c in feats]].rename(columns={c+"__wcm":c   for c in feats}).reset_index(drop=True)
nsqip_view = m[[c+"__nsqip" for c in feats]].rename(columns={c+"__nsqip":c for c in feats}).reset_index(drop=True)

# CPT arrays for one-hot (each view uses its own PrimaryCPT which will be near-identical since it's the same case)
cpt_wcm   = wcm_view["PrimaryCPT"].astype(str).to_numpy().reshape(-1,1)
cpt_nsqip = nsqip_view["PrimaryCPT"].astype(str).to_numpy().reshape(-1,1)

# ------------------------------------------------------------
# 2. Models A/B — RF on tabular
# ------------------------------------------------------------
def _bootstrap(y_true, p, fn, seed):
    return _bootstrap_ci(y_true, p, fn, n_boot=2000, seed=seed)

def rf_cv(view, cpt, label):
    N = len(view); p = np.full(N, np.nan)
    skf = StratifiedKFold(K, shuffle=True, random_state=SEED)
    feat_use = [c for c in view.columns if c != "PrimaryCPT"]
    for tr, te in skf.split(np.zeros(N), y):
        _, st = fit_preprocess(view[feat_use].iloc[tr].reset_index(drop=True), id_cols=[])
        X = apply_preprocess(view[feat_use], st)
        oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(cpt[tr])
        Xd = np.hstack([X, oh.transform(cpt)])
        base = RandomForestClassifier(500, min_samples_leaf=10, max_features="sqrt",
                                       class_weight="balanced", random_state=42, n_jobs=-1)
        est = CalibratedClassifierCV(base, method="isotonic", cv=3).fit(Xd[tr], y[tr])
        p[te] = est.predict_proba(Xd[te])[:, 1]
    au = roc_auc_score(y, p); ap = average_precision_score(y, p); br = brier_score_loss(y, p)
    au_ci = _bootstrap(y, p, roc_auc_score, 0)
    ap_ci = _bootstrap(y, p, average_precision_score, 1)
    print(f"  {label:35s} AUROC {au:.3f} ({au_ci[0]:.3f}-{au_ci[1]:.3f})  AUPRC {ap:.3f} ({ap_ci[0]:.3f}-{ap_ci[1]:.3f})  Brier {br:.4f}", flush=True)
    return dict(name=label, auroc=au, auroc_lo=au_ci[0], auroc_hi=au_ci[1],
                auprc=ap, auprc_lo=ap_ci[0], auprc_hi=ap_ci[1], brier=br, oof=p)

print("\n=== RF baselines on the matched cohort (5-fold OOF) ===", flush=True)
rA = rf_cv(nsqip_view, cpt_nsqip, "A. RF on NSQIP features")
rB = rf_cv(wcm_view,   cpt_wcm,   "B. RF on WCM features")

# ------------------------------------------------------------
# 3. Build the WCM heterograph, then restrict to matched encounters
# ------------------------------------------------------------
print("\n[t] loading raw graph tables + building trajectory/calendar features on FULL cohort...", flush=True)
raw_full = load_raw()

# Full cohort merged (like cv_compare)
enc_features_no_dupes = raw_full.enc_features.drop(
    columns=([c for c in raw_full.encounters.columns
              if c != "LogID" and c in raw_full.enc_features.columns]
             + ["ReadmittedWithin30Days"]), errors="ignore")
merged_full = (raw_full.encounters
               .merge(enc_features_no_dupes, on="LogID", how="inner")
               .merge(raw_full.labels[["LogID","ReadmittedWithin30Days"]], on="LogID", how="inner")
               .reset_index(drop=True))
merged_full["LogID"] = merged_full["LogID"].astype(str)
ss = merged_full[["LogID"]].copy()
ss["_ss"] = pd.to_datetime(merged_full.get("Procedure/Surgery Start"), errors="coerce")
traj = build_preop_trajectory_features(raw_full.enc_unit_edges, ss)
merged_full = merged_full.merge(traj, on="LogID", how="left")
for c in C.TRAJECTORY_FEATURE_COLUMNS:
    merged_full[c] = merged_full[c].fillna(0)
merged_full = add_calendar_features(merged_full)

nsqip_cols   = [c for c in C.MODEL_FEATURE_COLUMNS if c in merged_full.columns]
derived_cols = [c for c in C.ENCOUNTER_DERIVED_COLUMNS if c in merged_full.columns]
graph_feat_cols = nsqip_cols + derived_cols
print(f"[t] full cohort N={len(merged_full)}   graph feat cols n={len(graph_feat_cols)}", flush=True)

# indices of matched LogIDs within merged_full (in the order we've been using: m)
logid_to_full = {lid: i for i, lid in enumerate(merged_full["LogID"].tolist())}
m_logids = m["LogID"].astype(str).values
matched_full_idx = np.array([logid_to_full[l] for l in m_logids if l in logid_to_full])
matched_valid = np.array([l in logid_to_full for l in m_logids])
if not matched_valid.all():
    print(f"[t][warn] {int((~matched_valid).sum())} matched LogIDs not found in full graph cohort — dropping", flush=True)
    m2 = m.loc[matched_valid].reset_index(drop=True)
    y = m2["y_nsqip"].astype(int).values
    nsqip_view = nsqip_view.loc[matched_valid].reset_index(drop=True)
    wcm_view = wcm_view.loc[matched_valid].reset_index(drop=True)
    cpt_wcm = wcm_view["PrimaryCPT"].astype(str).to_numpy().reshape(-1,1)
    cpt_nsqip = nsqip_view["PrimaryCPT"].astype(str).to_numpy().reshape(-1,1)
    matched_full_idx = np.array([logid_to_full[l] for l in m2["LogID"].astype(str).values])

print(f"[t] usable matched rows (in-full-cohort): {len(matched_full_idx)}", flush=True)

# provider/unit node features (unchanged from full cohort)
prov_ids, X_prov, _ = build_provider_features(raw_full.prov_attrs)
unit_ids, X_unit, _ = build_unit_features(raw_full.unit_attrs)

# The MedHG-PS graph is built on the FULL cohort. We'll train the ie-HGCN on the
# full graph but only score / mask on the 1,583 matched encounters.
# To swap WCM feature values for NSQIP feature values for the matched encounters, we
# build two views of X_enc: base = merged_full's own feat_cols; nsqip_override =
# same, but with the matched-encounter rows replaced with NSQIP values.

# Build NSQIP-override feature array aligned to merged_full's feat_col schema
nsqip_view_reindexed = pd.DataFrame(index=range(len(merged_full)), columns=graph_feat_cols, dtype=object)
for i, midx in enumerate(matched_full_idx):
    row_ns = nsqip_view.iloc[i]
    for col in graph_feat_cols:
        if col in row_ns.index and pd.notna(row_ns[col]):
            nsqip_view_reindexed.loc[midx, col] = row_ns[col]
        else:
            nsqip_view_reindexed.loc[midx, col] = merged_full.iloc[midx].get(col, np.nan)
# For non-matched rows, keep WCM values (so the graph training still has data)
for midx in set(range(len(merged_full))) - set(matched_full_idx.tolist()):
    for col in graph_feat_cols:
        nsqip_view_reindexed.loc[midx, col] = merged_full.iloc[midx].get(col, np.nan)

# ------------------------------------------------------------
# 4. GNN CV loop: models C, D, E
# ------------------------------------------------------------
dev = _resolve_device(C.DEFAULTS_TRAIN.device)
print(f"\n[t] training ie-HGCN on full graph; scoring only matched encounters. device={dev}", flush=True)

# labels for the graph training use WCM (full-cohort) labels but we'll only
# EVALUATE the matched-encounter subset against y (NSQIP gold outcome).
# To keep things clean, train the graph with y_wcm as target on the full cohort,
# then evaluate its predictions on matched encounters against y (NSQIP).
# This is a compromise; alternative (train only on matched with masked graph) is
# too small (~1500 nodes) for the GNN.
y_full = merged_full["ReadmittedWithin30Days"].astype(int).values
N_full = len(merged_full)

# Map matched encounter -> its position in y (nsqip outcome)
# Build a NSQIP-outcome vector aligned to merged_full, using WCM label as fallback
# for non-matched rows (only matched rows will be scored anyway).
y_gold_aligned = y_full.copy()
for i, midx in enumerate(matched_full_idx):
    y_gold_aligned[midx] = int(y[i])

def train_gnn_and_get_matched_preds(enc_feat_source_df, label_source_arr,
                                    tag, seed_offset=0):
    """Train ie-HGCN once (full graph), split into folds by MATCHED-encounter subset,
    return OOF probs on matched encounters only.

    We do 5-fold CV on the matched subset. In each fold, we train the GNN on the
    graph where the test-fold matched encounters are masked out (no label leakage)
    plus all non-matched encounters as fixed extra training data.
    """
    p_matched = np.full(len(matched_full_idx), np.nan)
    skf = StratifiedKFold(K, shuffle=True, random_state=SEED)
    matched_idx_arr = np.array(matched_full_idx)
    y_matched = np.array([y_full[i] for i in matched_idx_arr])  # not used for stratifying — use y (nsqip)
    for fold, (tr_local, te_local) in enumerate(skf.split(np.zeros(len(matched_idx_arr)), y)):
        rng = np.random.default_rng(SEED + seed_offset + fold)
        tr_local = tr_local.copy(); rng.shuffle(tr_local)
        n_val = int(round(VAL_FRAC * len(matched_idx_arr)))
        val_local, tr_local = tr_local[:n_val], tr_local[n_val:]

        train_mask = np.zeros(N_full, bool)
        val_mask   = np.zeros(N_full, bool)
        test_mask  = np.zeros(N_full, bool)
        # matched-cohort test rows
        test_full = matched_idx_arr[te_local]
        val_full  = matched_idx_arr[val_local]
        tr_full   = matched_idx_arr[tr_local]
        train_mask[tr_full] = True
        # ALSO add all non-matched encounters to train (they don't overlap the eval)
        non_matched = np.setdiff1d(np.arange(N_full), matched_idx_arr)
        train_mask[non_matched] = True
        val_mask[val_full] = True
        test_mask[test_full] = True

        # fit preprocessing on train rows only (this fold)
        feat_all = enc_feat_source_df[graph_feat_cols].copy()
        _, enc_state = fit_preprocess(feat_all.loc[train_mask].reset_index(drop=True), id_cols=[])
        X_enc = apply_preprocess(feat_all, enc_state)

        artifacts = build_graph(
            raw=raw_full, encounters_merged=merged_full, enc_features=X_enc,
            prov_ids=prov_ids, prov_features=X_prov,
            unit_ids=unit_ids, unit_features=X_unit,
            train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
        )
        set_seed(SEED + seed_offset + fold)
        # label the graph with y_gold_aligned so matched encounters have NSQIP outcome
        # (this only matters for the training loss on the matched subset).
        artifacts.g.nodes["encounter"].data["label"] = torch.tensor(label_source_arr, dtype=torch.long)
        model, _ = train_model(artifacts, cfg=C.DEFAULTS_TRAIN, save_dir=None, verbose=False)
        model.eval(); g_dev = artifacts.g.to(dev); model = model.to(dev)
        with torch.no_grad():
            logits, hs = model(g_dev, {nt: g_dev.nodes[nt].data["h"] for nt in C.NODE_TYPES})
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        p_matched[te_local] = probs[test_full]
        print(f"  [{tag}] fold {fold+1}/{K}: matched-test AUROC (raw GNN)={roc_auc_score(y[te_local], probs[test_full]):.3f}", flush=True)

    return p_matched, model, artifacts  # return last model/artifacts for embedding extract in Model E

def report_gnn(name, p):
    au = roc_auc_score(y, p); ap = average_precision_score(y, p); br = brier_score_loss(y, p)
    au_ci = _bootstrap(y, p, roc_auc_score, 0)
    ap_ci = _bootstrap(y, p, average_precision_score, 1)
    print(f"  {name:40s} AUROC {au:.3f} ({au_ci[0]:.3f}-{au_ci[1]:.3f})  AUPRC {ap:.3f} ({ap_ci[0]:.3f}-{ap_ci[1]:.3f})  Brier {br:.4f}", flush=True)
    return dict(name=name, auroc=au, auroc_lo=au_ci[0], auroc_hi=au_ci[1],
                auprc=ap, auprc_lo=ap_ci[0], auprc_hi=ap_ci[1], brier=br, oof=p)

# Model C: ie-HGCN with NSQIP features on matched encounters
print("\n=== Model C: ie-HGCN raw with NSQIP encounter features ===", flush=True)
pC, modC, artC = train_gnn_and_get_matched_preds(nsqip_view_reindexed, y_gold_aligned, "C.NSQIP", seed_offset=0)
rC = report_gnn("C. ie-HGCN raw (NSQIP feats)", pC)

# Model D: ie-HGCN with WCM features on all encounters (unchanged features)
print("\n=== Model D: ie-HGCN raw with WCM encounter features ===", flush=True)
pD, modD, artD = train_gnn_and_get_matched_preds(merged_full, y_gold_aligned, "D.WCM", seed_offset=100)
rD = report_gnn("D. ie-HGCN raw (WCM feats)", pD)

# Model E: NSQIP RF stacked with ie-HGCN encounter embedding (using NSQIP model)
print("\n=== Model E: NSQIP RF + ie-HGCN encounter embedding stack ===", flush=True)
# Extract encounter embeddings from the last-fold Model-C GNN
tables = extract_embeddings(modC, artC, raw_prov_attrs=raw_full.prov_attrs,
                            raw_unit_attrs=raw_full.unit_attrs, device=dev)
enc_emb = tables.encounter.copy()
enc_emb["LogID"] = enc_emb["LogID"].astype(str)
emb_cols = [c for c in enc_emb.columns if c.startswith("emb_")]
emb_matched = pd.DataFrame(index=range(len(matched_full_idx)))
# align embeddings to matched-cohort order
enc_emb_indexed = enc_emb.set_index("LogID")
matched_logids = [merged_full.iloc[i]["LogID"] for i in matched_full_idx]
emb_matched = enc_emb_indexed.loc[matched_logids, emb_cols].reset_index(drop=True)

def rf_cv_with_stack(view, cpt, stack_arr, label):
    N = len(view); p = np.full(N, np.nan)
    skf = StratifiedKFold(K, shuffle=True, random_state=SEED)
    feat_use = [c for c in view.columns if c != "PrimaryCPT"]
    for tr, te in skf.split(np.zeros(N), y):
        _, st = fit_preprocess(view[feat_use].iloc[tr].reset_index(drop=True), id_cols=[])
        X = apply_preprocess(view[feat_use], st)
        oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(cpt[tr])
        Xd = np.hstack([X, oh.transform(cpt), stack_arr])
        base = RandomForestClassifier(500, min_samples_leaf=10, max_features="sqrt",
                                       class_weight="balanced", random_state=42, n_jobs=-1)
        est = CalibratedClassifierCV(base, method="isotonic", cv=3).fit(Xd[tr], y[tr])
        p[te] = est.predict_proba(Xd[te])[:, 1]
    return report_gnn(label, p)

rE = rf_cv_with_stack(nsqip_view, cpt_nsqip, emb_matched.values.astype(np.float32),
                     "E. NSQIP RF + ie-HGCN emb stack")

# ------------------------------------------------------------
# 5. Summary + save
# ------------------------------------------------------------
print("\n=== FINAL COMPARISON TABLE (matched cohort N={}, gold base {:.2f}%) ===".format(len(y), y.mean()*100), flush=True)
print(f"  {'model':45s} {'AUROC (95% CI)':22s} {'AUPRC (95% CI)':22s} {'Brier':8s} {'dAUROC':>8s} {'dAUPRC':>8s}", flush=True)
base_au = rA['auroc']; base_ap = rA['auprc']
for r in [rA, rB, rC, rD, rE]:
    print(f"  {r['name']:45s} {r['auroc']:.3f} ({r['auroc_lo']:.3f}-{r['auroc_hi']:.3f}){'':2s}"
          f"{r['auprc']:.3f} ({r['auprc_lo']:.3f}-{r['auprc_hi']:.3f}){'':2s}"
          f"{r['brier']:.4f}   {r['auroc']-base_au:+.4f}  {r['auprc']-base_ap:+.4f}", flush=True)

# Save
out_dir = Path("artifacts/newdata"); out_dir.mkdir(parents=True, exist_ok=True)
rows = [{k: v for k, v in r.items() if k != "oof"} for r in [rA, rB, rC, rD, rE]]
pd.DataFrame(rows).to_csv(out_dir / "medhgps_nsqip_matched_results.csv", index=False)
print(f"\n[t] saved -> {out_dir / 'medhgps_nsqip_matched_results.csv'}", flush=True)

print(f"\n=== Reference: WCM full-cohort RF (gold label, N=14,009, base 7.54%): AUROC 0.721 / AUPRC 0.172 ===", flush=True)
print("DONE", flush=True)
