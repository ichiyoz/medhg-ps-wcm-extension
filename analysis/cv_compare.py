"""Fold-honest K-fold CV: does the GNN beat a tabular baseline, robustly?

For each fold we RETRAIN the GNN on that fold's train set (embeddings can't be
reused across folds without leaking), extract encounter embeddings, and compare
three models on the held-out fold, all keyed to ONE frame so every model sees
the identical split:

  - gnn_raw     : the GNN's own softmax head
  - rf_tab      : RandomForest on the 40 NSQIP pre-op features
  - rf_tab_enc  : RandomForest on tabular + the GNN encounter embedding

Reports per-fold and mean +/- std AUROC/AUPRC, plus the paired
(rf_tab_enc - rf_tab) lift per fold. Reuses the pipeline's own building blocks
(assembly mirrors run.py). Writes nothing to disk (no artifact clobbering).

    PYTHONPATH=. python analysis/cv_compare.py
"""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import medhg_ps.config as C
from medhg_ps.data import (
    load_raw, fit_preprocess, apply_preprocess,
    build_provider_features, build_unit_features,
    build_preop_trajectory_features, add_calendar_features,
)
from medhg_ps.graph import build_graph
from medhg_ps.train import train_model, set_seed, _resolve_device
from medhg_ps.extract_embeddings import extract_embeddings
from medhg_ps.scripts.train_rf import _build_pipe

K = 5
SEED = 42
VAL_FRAC = 0.10

# ---- assemble merged_all ONCE (mirrors run.py) ----------------------
print("[cv] loading + assembling merged frame...", flush=True)
raw = load_raw()
enc_features_no_dupes = raw.enc_features.drop(
    columns=([c for c in raw.encounters.columns
              if c != "LogID" and c in raw.enc_features.columns]
             + ["ReadmittedWithin30Days"]),
    errors="ignore",
)
merged_all = (raw.encounters
              .merge(enc_features_no_dupes, on="LogID", how="inner")
              .merge(raw.labels[["LogID", "ReadmittedWithin30Days"]],
                     on="LogID", how="inner")
              .reset_index(drop=True))
ss = merged_all[["LogID"]].copy()
ss["_ss"] = pd.to_datetime(merged_all.get("Procedure/Surgery Start"), errors="coerce")
traj = build_preop_trajectory_features(raw.enc_unit_edges, ss)
merged_all = merged_all.merge(traj, on="LogID", how="left")
for c in C.TRAJECTORY_FEATURE_COLUMNS:
    merged_all[c] = merged_all[c].fillna(0)
merged_all = add_calendar_features(merged_all)

nsqip_cols   = [c for c in C.MODEL_FEATURE_COLUMNS if c in merged_all.columns]
derived_cols = [c for c in C.ENCOUNTER_DERIVED_COLUMNS if c in merged_all.columns]
feat_cols    = nsqip_cols + derived_cols
y = merged_all["ReadmittedWithin30Days"].astype(int).values
N = len(merged_all)
print(f"[cv] cohort={N:,}  base rate={y.mean()*100:.2f}%  "
      f"tabular feats={len(nsqip_cols)}  +derived={len(derived_cols)}", flush=True)

# provider/unit node features do not depend on the encounter split
prov_ids, X_prov, _ = build_provider_features(raw.prov_attrs)
unit_ids, X_unit, _ = build_unit_features(raw.unit_attrs)

dev = _resolve_device(C.DEFAULTS_TRAIN.device)
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
MODELS = ["gnn_raw", "rf_tab", "rf_tab_enc"]
res = {m: {"auroc": [], "auprc": []} for m in MODELS}
lift = {"auroc": [], "auprc": []}


def rf_metrics(Xcols, tr, te):
    X = merged_all[Xcols] if isinstance(Xcols, list) else Xcols
    pipe = _build_pipe(X)
    pipe.fit(X.iloc[tr], y[tr])
    p = pipe.predict_proba(X.iloc[te])[:, 1]
    return roc_auc_score(y[te], p), average_precision_score(y[te], p)


for fold, (tr_all, te) in enumerate(skf.split(np.zeros(N), y)):
    # carve a validation slice out of train (for GNN early stopping)
    rng = np.random.default_rng(SEED + fold)
    tr_all = tr_all.copy(); rng.shuffle(tr_all)
    n_val = int(round(VAL_FRAC * N))
    val, tr = tr_all[:n_val], tr_all[n_val:]
    train_mask = np.zeros(N, bool); train_mask[tr] = True
    val_mask   = np.zeros(N, bool); val_mask[val] = True
    test_mask  = np.zeros(N, bool); test_mask[te] = True

    # encounter features: fit preprocessing on THIS fold's train only
    feat_all = merged_all[feat_cols].copy()
    _, enc_state = fit_preprocess(feat_all.loc[train_mask].reset_index(drop=True), id_cols=[])
    X_enc = apply_preprocess(feat_all, enc_state)

    artifacts = build_graph(
        raw=raw, encounters_merged=merged_all, enc_features=X_enc,
        prov_ids=prov_ids, prov_features=X_prov,
        unit_ids=unit_ids, unit_features=X_unit,
        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
    )

    set_seed(SEED + fold)
    model, _ = train_model(artifacts, cfg=C.DEFAULTS_TRAIN, save_dir=None, verbose=False)

    # raw GNN head on the held-out fold
    model.eval()
    g_dev = artifacts.g.to(dev)
    with torch.no_grad():
        logits, _ = model.to(dev)(g_dev, {nt: g_dev.nodes[nt].data["h"] for nt in C.NODE_TYPES})
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    res["gnn_raw"]["auroc"].append(roc_auc_score(y[te], probs[te]))
    res["gnn_raw"]["auprc"].append(average_precision_score(y[te], probs[te]))

    # encounter embeddings -> RF
    tables = extract_embeddings(model, artifacts,
                                raw_prov_attrs=raw.prov_attrs,
                                raw_unit_attrs=raw.unit_attrs, device=dev)
    enc_emb = tables.encounter.copy()
    enc_emb["LogID"] = enc_emb["LogID"].astype(str)
    emb_cols = [c for c in enc_emb.columns if c.startswith("emb_")]
    frame = merged_all[["LogID"] + feat_cols].copy()
    frame["LogID"] = frame["LogID"].astype(str)
    frame = frame.merge(enc_emb, on="LogID", how="left")

    a_tab, p_tab = rf_metrics(feat_cols, tr, te)            # uses merged_all
    a_enc, p_enc = rf_metrics(frame[feat_cols + emb_cols], tr, te)
    res["rf_tab"]["auroc"].append(a_tab);     res["rf_tab"]["auprc"].append(p_tab)
    res["rf_tab_enc"]["auroc"].append(a_enc); res["rf_tab_enc"]["auprc"].append(p_enc)
    lift["auroc"].append(a_enc - a_tab); lift["auprc"].append(p_enc - p_tab)

    print(f"[cv] fold {fold+1}/{K}  "
          f"gnn_raw AUROC={res['gnn_raw']['auroc'][-1]:.3f}  "
          f"rf_tab={a_tab:.3f}  rf_tab_enc={a_enc:.3f}  "
          f"(enc-tab AUROC {a_enc-a_tab:+.3f} / AUPRC {p_enc-p_tab:+.3f})", flush=True)

# ---- aggregate ------------------------------------------------------
print(f"\n=== {K}-fold CV (mean +/- std) | base rate {y.mean()*100:.2f}% ===")
print(f"  {'model':12s} {'AUROC':>16s} {'AUPRC':>16s}")
for m in MODELS:
    au, ap = np.array(res[m]["auroc"]), np.array(res[m]["auprc"])
    print(f"  {m:12s} {au.mean():.3f} +/- {au.std():.3f}    "
          f"{ap.mean():.3f} +/- {ap.std():.3f}")

la, lp = np.array(lift["auroc"]), np.array(lift["auprc"])
print(f"\n=== Paired lift  rf_tab_enc - rf_tab  (per-fold) ===")
print(f"  dAUROC mean {la.mean():+.4f} +/- {la.std():.4f}  "
      f"(folds>0: {int((la>0).sum())}/{K})  per-fold {np.round(la,3).tolist()}")
print(f"  dAUPRC mean {lp.mean():+.4f} +/- {lp.std():.4f}  "
      f"(folds>0: {int((lp>0).sum())}/{K})  per-fold {np.round(lp,3).tolist()}")
