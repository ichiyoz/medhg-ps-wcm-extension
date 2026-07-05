"""CV of the embedding-integration variants vs fair tabular baselines.

Per identical fold, retrain the GNN once, then build every integration path:
  - gnn_raw        : GNN softmax head
  - gnn_enc        : RF on tabular + encounter embedding              (variant 2)
  - gnn_prov_unit  : RF on tabular + mean-prov-emb + hrs-wtd unit-emb (variant 3)
  - rf_tab / hgb_tab : tabular baselines (RandomForest / HistGradientBoosting)

Aggregations mirror scripts/train_rf.py but run in-memory (fold-honest).
Reuses run.py's assembly; writes nothing.

    PYTHONPATH=. python analysis/cv_variant3.py
"""
from dataclasses import replace

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import medhg_ps.config as C
from medhg_ps.data import (
    load_raw, fit_preprocess, apply_preprocess,
    build_provider_features, build_unit_features,
    build_preop_trajectory_features, add_calendar_features,
)
from medhg_ps.graph import build_graph
from medhg_ps.train import train_model, set_seed, _resolve_device
from medhg_ps.extract_embeddings import extract_embeddings

K, SEED, VAL_FRAC = 5, 42, 0.10
GNN_CFG = replace(C.DEFAULTS_TRAIN, learning_rate=0.008, max_epochs=400, early_stop_patience=30)


def _norm(s):
    return s.astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)


# ---- assemble merged_all ONCE (mirrors run.py) ----------------------
raw = load_raw()
enc_features_no_dupes = raw.enc_features.drop(
    columns=([c for c in raw.encounters.columns
              if c != "LogID" and c in raw.enc_features.columns]
             + ["ReadmittedWithin30Days"]), errors="ignore")
merged_all = (raw.encounters
              .merge(enc_features_no_dupes, on="LogID", how="inner")
              .merge(raw.labels[["LogID", "ReadmittedWithin30Days"]],
                     on="LogID", how="inner").reset_index(drop=True))
ss = merged_all[["LogID"]].copy()
ss["_ss"] = pd.to_datetime(merged_all.get("Procedure/Surgery Start"), errors="coerce")
traj = build_preop_trajectory_features(raw.enc_unit_edges, ss)
merged_all = merged_all.merge(traj, on="LogID", how="left")
for c in C.TRAJECTORY_FEATURE_COLUMNS:
    merged_all[c] = merged_all[c].fillna(0)
merged_all = add_calendar_features(merged_all)
merged_all["LogID"] = merged_all["LogID"].astype(str)

nsqip_cols   = [c for c in C.MODEL_FEATURE_COLUMNS if c in merged_all.columns]
derived_cols = [c for c in C.ENCOUNTER_DERIVED_COLUMNS if c in merged_all.columns]
feat_cols = nsqip_cols + derived_cols
y = merged_all["ReadmittedWithin30Days"].astype(int).values
N = len(merged_all)
print(f"[cv] cohort={N:,}  base={y.mean()*100:.2f}%  feats={len(feat_cols)}", flush=True)

prov_ids, X_prov, _ = build_provider_features(raw.prov_attrs)
unit_ids, X_unit, _ = build_unit_features(raw.unit_attrs)
dev = _resolve_device(C.DEFAULTS_TRAIN.device)

# edge tables for aggregation
A2 = raw.enc_prov_edges[["LogID", "ProvID"]].copy()
A2["LogID"] = _norm(A2["LogID"]); A2["ProvID"] = _norm(A2["ProvID"])
A3 = raw.enc_unit_edges[["LogID", "DepartmentID", "Hours"]].copy()
A3["LogID"] = _norm(A3["LogID"]); A3["DepartmentID"] = _norm(A3["DepartmentID"])


def agg_prov(prov_tbl):
    cols = [c for c in prov_tbl.columns if c.startswith("emb_")]
    p = prov_tbl.copy(); p["ProvID"] = _norm(p["ProvID"])
    j = A2.merge(p, on="ProvID", how="left").dropna(subset=cols, how="all")
    g = j.groupby("LogID")[cols].mean().reset_index()
    g.columns = ["LogID"] + [f"prov_{c}" for c in cols]
    return g


def agg_unit(unit_tbl):
    cols = [c for c in unit_tbl.columns if c.startswith("emb_")]
    u = unit_tbl.copy(); u["DepartmentID"] = _norm(u["DepartmentID"])
    j = A3.merge(u, on="DepartmentID", how="left")
    j["Hours"] = pd.to_numeric(j["Hours"], errors="coerce").fillna(0).clip(lower=0)
    rows = []
    for lid, gg in j.groupby("LogID"):
        w = gg["Hours"].values; em = gg[cols].values
        if w.sum() <= 0 or np.all(np.isnan(em)):
            valid = ~np.isnan(em).all(axis=1)
            avg = np.nanmean(em[valid], axis=0) if valid.any() else np.zeros(len(cols))
        else:
            avg = np.where(np.isnan(em), 0.0, em).T @ (w / w.sum())
        rows.append([lid] + list(avg))
    return pd.DataFrame(rows, columns=["LogID"] + [f"unit_{c}" for c in cols])


def _pre(X):
    cat = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    return ColumnTransformer([
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Unknown")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
        ("num", Pipeline([("imp", SimpleImputer(strategy="mean")),
                          ("sca", StandardScaler())]), num)])


def fit_eval(est, X, tr, te):
    pipe = Pipeline([("pre", _pre(X)), ("clf", est)])
    pipe.fit(X.iloc[tr], y[tr])
    p = pipe.predict_proba(X.iloc[te])[:, 1]
    return roc_auc_score(y[te], p), average_precision_score(y[te], p)


def _rf():
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=10, max_features="sqrt",
                                  class_weight="balanced", random_state=42, n_jobs=-1)


def _hgb():
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                          l2_regularization=1.0, random_state=42)


skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
MODELS = ["gnn_raw", "gnn_enc", "gnn_prov_unit", "rf_tab", "hgb_tab"]
res = {m: {"auroc": [], "auprc": []} for m in MODELS}

for fold, (tr_all, te) in enumerate(skf.split(np.zeros(N), y)):
    rng = np.random.default_rng(SEED + fold)
    tr_all = tr_all.copy(); rng.shuffle(tr_all)
    n_val = int(round(VAL_FRAC * N))
    val, tr = tr_all[:n_val], tr_all[n_val:]
    tm = np.zeros(N, bool); tm[tr] = True
    vm = np.zeros(N, bool); vm[val] = True
    em = np.zeros(N, bool); em[te] = True

    feat_all = merged_all[feat_cols].copy()
    _, st = fit_preprocess(feat_all.loc[tm].reset_index(drop=True), id_cols=[])
    X_enc = apply_preprocess(feat_all, st)
    artifacts = build_graph(raw=raw, encounters_merged=merged_all, enc_features=X_enc,
                            prov_ids=prov_ids, prov_features=X_prov,
                            unit_ids=unit_ids, unit_features=X_unit,
                            train_mask=tm, val_mask=vm, test_mask=em)
    set_seed(SEED + fold)
    model, trr = train_model(artifacts, cfg=GNN_CFG, save_dir=None, verbose=False)

    model.eval()
    g_dev = artifacts.g.to(dev)
    with torch.no_grad():
        logits, _ = model.to(dev)(g_dev, {nt: g_dev.nodes[nt].data["h"] for nt in C.NODE_TYPES})
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    res["gnn_raw"]["auroc"].append(roc_auc_score(y[te], probs[te]))
    res["gnn_raw"]["auprc"].append(average_precision_score(y[te], probs[te]))

    tab = extract_embeddings(model, artifacts, raw_prov_attrs=raw.prov_attrs,
                             raw_unit_attrs=raw.unit_attrs, device=dev)
    enc = tab.encounter.copy(); enc["LogID"] = _norm(enc["LogID"])
    ecols = [c for c in enc.columns if c.startswith("emb_")]

    f_enc = merged_all[["LogID"] + feat_cols].merge(enc, on="LogID", how="left")
    f_pu = (merged_all[["LogID"] + feat_cols]
            .merge(agg_prov(tab.provider), on="LogID", how="left")
            .merge(agg_unit(tab.unit), on="LogID", how="left"))
    pu_cols = [c for c in f_pu.columns if c.startswith(("prov_emb_", "unit_emb_"))]

    a_e, p_e = fit_eval(_rf(), f_enc[feat_cols + ecols], tr, te)
    a_p, p_p = fit_eval(_rf(), f_pu[feat_cols + pu_cols], tr, te)
    a_r, p_r = fit_eval(_rf(), merged_all[feat_cols], tr, te)
    a_h, p_h = fit_eval(_hgb(), merged_all[feat_cols], tr, te)
    res["gnn_enc"]["auroc"].append(a_e);       res["gnn_enc"]["auprc"].append(p_e)
    res["gnn_prov_unit"]["auroc"].append(a_p); res["gnn_prov_unit"]["auprc"].append(p_p)
    res["rf_tab"]["auroc"].append(a_r);        res["rf_tab"]["auprc"].append(p_r)
    res["hgb_tab"]["auroc"].append(a_h);       res["hgb_tab"]["auprc"].append(p_h)
    bt = max(a_r, a_h)
    print(f"[cv] fold {fold+1}/{K} (stop@{trr.best_epoch})  enc={a_e:.3f} "
          f"prov_unit={a_p:.3f}(AUPRC {p_p:.3f})  rf={a_r:.3f} hgb={a_h:.3f}  "
          f"(prov_unit-best_tab {a_p-bt:+.3f})", flush=True)

print(f"\n=== {K}-fold CV (mean +/- std) | base {y.mean()*100:.2f}% ===")
print(f"  {'model':14s} {'AUROC':>16s} {'AUPRC':>16s}")
for m in MODELS:
    au, ap = np.array(res[m]["auroc"]), np.array(res[m]["auprc"])
    print(f"  {m:14s} {au.mean():.3f} +/- {au.std():.3f}    {ap.mean():.3f} +/- {ap.std():.3f}")
bt_au = np.maximum(res["rf_tab"]["auroc"], res["hgb_tab"]["auroc"])
bt_ap = np.maximum(res["rf_tab"]["auprc"], res["hgb_tab"]["auprc"])
for m in ["gnn_enc", "gnn_prov_unit"]:
    da = np.array(res[m]["auroc"]) - bt_au
    dp = np.array(res[m]["auprc"]) - bt_ap
    print(f"\n  {m} - best_tab:  dAUROC {da.mean():+.4f} (folds>0 {int((da>0).sum())}/{K}) | "
          f"dAUPRC {dp.mean():+.4f} (folds>0 {int((dp>0).sum())}/{K})")
