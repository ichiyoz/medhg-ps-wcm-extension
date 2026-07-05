"""CV check: does a RETUNED GNN beat a (fair) tabular baseline?

Tests the hypothesis that the GNN underperformed only because the full-batch
regime was under-trained at lr=0.001. Bumps lr/epochs and CV-compares, per
identical fold:
  - gnn_raw   : retuned GNN softmax head
  - gnn_enc   : RF on tabular + retuned-GNN encounter embedding
  - rf_tab    : RandomForest tabular baseline
  - hgb_tab   : HistGradientBoosting tabular baseline (stronger; fair bar)

Reports mean +/- std AUROC/AUPRC and the paired lift of the GNN paths over the
BEST tabular baseline per fold. Reuses run.py's assembly; writes nothing.

    PYTHONPATH=. python analysis/cv_gnn_retuned.py
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
# Retuned for the full-batch regime (1 step/epoch): higher lr, more epochs,
# more patience so it doesn't stop while val AUROC is still creeping up.
GNN_CFG = replace(C.DEFAULTS_TRAIN, learning_rate=0.008,
                  max_epochs=400, early_stop_patience=30)

print(f"[cv] retuned GNN: lr={GNN_CFG.learning_rate} max_epochs={GNN_CFG.max_epochs} "
      f"patience={GNN_CFG.early_stop_patience}", flush=True)

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

nsqip_cols   = [c for c in C.MODEL_FEATURE_COLUMNS if c in merged_all.columns]
derived_cols = [c for c in C.ENCOUNTER_DERIVED_COLUMNS if c in merged_all.columns]
feat_cols = nsqip_cols + derived_cols
y = merged_all["ReadmittedWithin30Days"].astype(int).values
N = len(merged_all)
print(f"[cv] cohort={N:,}  base={y.mean()*100:.2f}%  feats={len(feat_cols)}", flush=True)

prov_ids, X_prov, _ = build_provider_features(raw.prov_attrs)
unit_ids, X_unit, _ = build_unit_features(raw.unit_attrs)
dev = _resolve_device(C.DEFAULTS_TRAIN.device)


def _pre(X):
    cat = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    return ColumnTransformer([
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Unknown")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
        ("num", Pipeline([("imp", SimpleImputer(strategy="mean")),
                          ("sca", StandardScaler())]), num),
    ])


def fit_eval(est, X, tr, te):
    pipe = Pipeline([("pre", _pre(X)), ("clf", est)])
    pipe.fit(X.iloc[tr], y[tr])
    p = pipe.predict_proba(X.iloc[te])[:, 1]
    return roc_auc_score(y[te], p), average_precision_score(y[te], p)


def _rf():
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=10,
                                  max_features="sqrt", class_weight="balanced",
                                  random_state=42, n_jobs=-1)


def _hgb():
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                          l2_regularization=1.0, random_state=42)


skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
MODELS = ["gnn_raw", "gnn_enc", "rf_tab", "hgb_tab"]
res = {m: {"auroc": [], "auprc": []} for m in MODELS}

for fold, (tr_all, te) in enumerate(skf.split(np.zeros(N), y)):
    rng = np.random.default_rng(SEED + fold)
    tr_all = tr_all.copy(); rng.shuffle(tr_all)
    n_val = int(round(VAL_FRAC * N))
    val, tr = tr_all[:n_val], tr_all[n_val:]
    train_mask = np.zeros(N, bool); train_mask[tr] = True
    val_mask   = np.zeros(N, bool); val_mask[val] = True
    test_mask  = np.zeros(N, bool); test_mask[te] = True

    feat_all = merged_all[feat_cols].copy()
    _, enc_state = fit_preprocess(feat_all.loc[train_mask].reset_index(drop=True), id_cols=[])
    X_enc = apply_preprocess(feat_all, enc_state)
    artifacts = build_graph(raw=raw, encounters_merged=merged_all, enc_features=X_enc,
                            prov_ids=prov_ids, prov_features=X_prov,
                            unit_ids=unit_ids, unit_features=X_unit,
                            train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    set_seed(SEED + fold)
    model, tr_res = train_model(artifacts, cfg=GNN_CFG, save_dir=None, verbose=False)

    model.eval()
    g_dev = artifacts.g.to(dev)
    with torch.no_grad():
        logits, _ = model.to(dev)(g_dev, {nt: g_dev.nodes[nt].data["h"] for nt in C.NODE_TYPES})
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    res["gnn_raw"]["auroc"].append(roc_auc_score(y[te], probs[te]))
    res["gnn_raw"]["auprc"].append(average_precision_score(y[te], probs[te]))

    tables = extract_embeddings(model, artifacts, raw_prov_attrs=raw.prov_attrs,
                                raw_unit_attrs=raw.unit_attrs, device=dev)
    enc_emb = tables.encounter.copy()
    enc_emb["LogID"] = enc_emb["LogID"].astype(str)
    emb_cols = [c for c in enc_emb.columns if c.startswith("emb_")]
    frame = merged_all[["LogID"] + feat_cols].copy()
    frame["LogID"] = frame["LogID"].astype(str)
    frame = frame.merge(enc_emb, on="LogID", how="left")

    a_enc, p_enc = fit_eval(_rf(), frame[feat_cols + emb_cols], tr, te)
    a_rf,  p_rf  = fit_eval(_rf(),  merged_all[feat_cols], tr, te)
    a_hg,  p_hg  = fit_eval(_hgb(), merged_all[feat_cols], tr, te)
    res["gnn_enc"]["auroc"].append(a_enc); res["gnn_enc"]["auprc"].append(p_enc)
    res["rf_tab"]["auroc"].append(a_rf);   res["rf_tab"]["auprc"].append(p_rf)
    res["hgb_tab"]["auroc"].append(a_hg);  res["hgb_tab"]["auprc"].append(p_hg)

    best_tab = max(a_rf, a_hg)
    print(f"[cv] fold {fold+1}/{K} (gnn stop@{tr_res.best_epoch})  "
          f"gnn_raw={res['gnn_raw']['auroc'][-1]:.3f}  gnn_enc={a_enc:.3f}  "
          f"rf_tab={a_rf:.3f}  hgb_tab={a_hg:.3f}  "
          f"(gnn_enc - best_tab AUROC {a_enc-best_tab:+.3f})", flush=True)

print(f"\n=== {K}-fold CV (mean +/- std) | base {y.mean()*100:.2f}% ===")
print(f"  {'model':10s} {'AUROC':>16s} {'AUPRC':>16s}")
for m in MODELS:
    au, ap = np.array(res[m]["auroc"]), np.array(res[m]["auprc"])
    print(f"  {m:10s} {au.mean():.3f} +/- {au.std():.3f}    {ap.mean():.3f} +/- {ap.std():.3f}")

best_tab_au = np.maximum(res["rf_tab"]["auroc"], res["hgb_tab"]["auroc"])
for gnn in ["gnn_raw", "gnn_enc"]:
    d = np.array(res[gnn]["auroc"]) - best_tab_au
    print(f"\n  {gnn} - best_tab  dAUROC mean {d.mean():+.4f} +/- {d.std():.4f} "
          f"(folds>0: {int((d>0).sum())}/{K})  per-fold {np.round(d,3).tolist()}")
