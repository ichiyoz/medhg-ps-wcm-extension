"""DICE on GNN embeddings: feed the ie-HGCN encounter embeddings (not raw
tabular features) into DICE, then run the DICE -> ML pipeline. Tests whether
clustering the graph representation (a) beats the tabular baseline / plain
tabular-DICE, and (b) stratifies risk differently -- esp. for the Z-code
(SDOH_Any_Z==1) subgroup.

Per fold the GNN is retrained on train rows only (mirrors cv_compare); encounter
embeddings E are extracted for all rows and standardized (scaler fit on train).
DICE is fit on E (K=4, d=16 -- NAS skipped for runtime); its soft memberships /
latent are fed downstream alongside the tabular block. Surrogate significance is
used in the CV loop to bound runtime (the GNN per-fold training dominates).

    PYTHONPATH=. python analysis/cv_dice_gnn.py
"""
from __future__ import annotations
import os
os.environ.setdefault("DICE_SURROGATE_SIG", "1")          # bound runtime (GNN dominates)

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import medhg_ps.config as C
from medhg_ps.data import (load_raw, fit_preprocess, apply_preprocess,
                           build_provider_features, build_unit_features,
                           build_preop_trajectory_features, add_calendar_features)
from medhg_ps.graph import build_graph
from medhg_ps.train import train_model, set_seed, _resolve_device
from medhg_ps.extract_embeddings import extract_embeddings
from medhg_ps.deploy import _load_cpt_map
import analysis.dice as dice

SMOKE = os.environ.get("MEDHG_SMOKE") == "1"
K_FOLDS, SEED, VAL_FRAC = (2 if SMOKE else 5), 42, 0.10
DK, DD = 4, 16                                             # fixed DICE K, latent d
HGB = lambda: HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                             l2_regularization=1.0, random_state=42)
LR = lambda: LogisticRegression(max_iter=1000, class_weight="balanced")

# ---- assemble merged_all ONCE (mirrors cv_compare) -----------------------
print(f"[dgnn] {'SMOKE ' if SMOKE else ''}loading...", flush=True)
raw = load_raw()
enc_nodupes = raw.enc_features.drop(
    columns=([c for c in raw.encounters.columns
              if c != "LogID" and c in raw.enc_features.columns] + ["ReadmittedWithin30Days"]),
    errors="ignore")
merged = (raw.encounters.merge(enc_nodupes, on="LogID", how="inner")
          .merge(raw.labels[["LogID", "ReadmittedWithin30Days"]], on="LogID", how="inner")
          .reset_index(drop=True))
ss = merged[["LogID"]].copy()
ss["_ss"] = pd.to_datetime(merged.get("Procedure/Surgery Start"), errors="coerce")
merged = merged.merge(build_preop_trajectory_features(raw.enc_unit_edges, ss), on="LogID", how="left")
for c in C.TRAJECTORY_FEATURE_COLUMNS:
    merged[c] = merged[c].fillna(0)
merged = add_calendar_features(merged)
merged["LogID"] = merged["LogID"].astype(str)

feat_cols = ([c for c in C.MODEL_FEATURE_COLUMNS if c in merged.columns]
             + [c for c in C.ENCOUNTER_DERIVED_COLUMNS if c in merged.columns])
y = merged["ReadmittedWithin30Days"].astype(int).values
N = len(merged)
cpt_map = _load_cpt_map()
cpt_arr = np.array([cpt_map.get(l, "UNK") for l in merged["LogID"]]).reshape(-1, 1)
anyz = pd.to_numeric(merged.get("SDOH_Any_Z", 0), errors="coerce").fillna(0).values.astype(int)
prov_ids, X_prov, _ = build_provider_features(raw.prov_attrs)
unit_ids, X_unit, _ = build_unit_features(raw.unit_attrs)
dev = _resolve_device(C.DEFAULTS_TRAIN.device)
print(f"[dgnn] cohort={N:,}  base={y.mean()*100:.2f}%  tab feats={len(feat_cols)}  "
      f"Z-code+={int(anyz.sum())}  surrogate_sig={os.environ.get('DICE_SURROGATE_SIG')}", flush=True)


def build_xtab(tr):
    _, st = fit_preprocess(merged[feat_cols].iloc[tr].reset_index(drop=True), id_cols=[])
    Xtab = apply_preprocess(merged[feat_cols], st)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(cpt_arr[tr])
    X = np.hstack([Xtab, ohe.transform(cpt_arr)])
    return StandardScaler().fit(X[tr]).transform(X)


def build_v(fit_rows):
    age = pd.to_numeric(merged["AgeYears"], errors="coerce").values.reshape(-1, 1)
    age = np.where(np.isnan(age), np.nanmedian(age[fit_rows]), age)
    age = (age - age[fit_rows].mean()) / (age[fit_rows].std() + 1e-8)
    g = merged[["Gender"]].astype(str)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(g.iloc[fit_rows])
    return np.hstack([age, ohe.transform(g)]).astype(np.float32)


def gnn_embeddings(tr, val):
    """Train GNN on train rows; return standardized encounter embeddings E for
    ALL rows (scaler fit on train). Mirrors cv_compare."""
    train_mask = np.zeros(N, bool); train_mask[tr] = True
    val_mask = np.zeros(N, bool); val_mask[val] = True
    test_mask = ~(train_mask | val_mask)
    _, enc_state = fit_preprocess(merged[feat_cols].loc[train_mask].reset_index(drop=True), id_cols=[])
    X_enc = apply_preprocess(merged[feat_cols], enc_state)
    artifacts = build_graph(raw=raw, encounters_merged=merged, enc_features=X_enc,
                            prov_ids=prov_ids, prov_features=X_prov,
                            unit_ids=unit_ids, unit_features=X_unit,
                            train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    set_seed(SEED)
    model, _ = train_model(artifacts, cfg=C.DEFAULTS_TRAIN, save_dir=None, verbose=False)
    tables = extract_embeddings(model, artifacts, raw_prov_attrs=raw.prov_attrs,
                                raw_unit_attrs=raw.unit_attrs, device=dev)
    enc = tables.encounter.copy(); enc["LogID"] = enc["LogID"].astype(str)
    emb_cols = [c for c in enc.columns if c.startswith("emb_")]
    E = (merged[["LogID"]].merge(enc[["LogID"] + emb_cols], on="LogID", how="left")
         .drop(columns="LogID").fillna(0.0).values.astype(np.float32))
    return StandardScaler().fit(E[tr]).transform(E)


def _tree(D, ftr, te):
    m = HGB().fit(D[ftr], y[ftr]); return m.predict_proba(D[te])[:, 1]


def _lr(D, ftr, te):
    sc = StandardScaler().fit(D[ftr]); m = LR().fit(sc.transform(D[ftr]), y[ftr])
    return m.predict_proba(sc.transform(D[te]))[:, 1]


def main():
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    models = ["tab", "gnn_enc_stack", "dice_gnn_only_lr", "dice_gnn_lr",
              "dice_gnn_gbdt", "dice_gnn_full_gbdt"]
    R = {m: {"au": [], "ap": []} for m in models}
    for fi, (tr_all, te) in enumerate(skf.split(np.zeros(N), y)):
        rng = np.random.default_rng(SEED + fi); tr_all = tr_all.copy(); rng.shuffle(tr_all)
        nv = int(round(VAL_FRAC * N)); val, tr = tr_all[:nv], tr_all[nv:]
        full_tr = np.concatenate([tr, val])

        E = gnn_embeddings(tr, val)                       # GNN encounter embeddings
        v = build_v(full_tr)
        dmodel = dice.fit(E[full_tr], y[full_tr], DK, DD, v[full_tr])
        chat = dice.cluster_proba(dmodel, E)              # [N, K]
        z = dice.embed(dmodel, E)                         # [N, d]
        X = build_xtab(tr)

        XD = np.hstack([chat, X]); XZ = np.hstack([z, chat, X]); XE = np.hstack([E, X])
        preds = {
            "tab":                _tree(X, full_tr, te),
            "gnn_enc_stack":      _tree(XE, full_tr, te),
            "dice_gnn_only_lr":   _lr(chat, full_tr, te),
            "dice_gnn_lr":        _lr(XD, full_tr, te),
            "dice_gnn_gbdt":      _tree(XD, full_tr, te),
            "dice_gnn_full_gbdt": _tree(XZ, full_tr, te),
        }
        for m in models:
            R[m]["au"].append(roc_auc_score(y[te], preds[m]))
            R[m]["ap"].append(average_precision_score(y[te], preds[m]))
        print(f"[dgnn] fold {fi+1}/{K_FOLDS} | "
              + "  ".join(f"{m} {R[m]['au'][-1]:.3f}" for m in models), flush=True)

    print(f"\n=== {K_FOLDS}-fold CV | base {y.mean()*100:.2f}% | DICE K={DK} d={DD} on GNN emb ===")
    print(f"  {'model':20s} {'AUROC':>16s} {'AUPRC':>16s}")
    for m in models:
        au, ap = np.array(R[m]["au"]), np.array(R[m]["ap"])
        print(f"  {m:20s} {au.mean():.3f} +/- {au.std():.3f}    {ap.mean():.3f} +/- {ap.std():.3f}")
    base = np.array(R["tab"]["au"]); base_ap = np.array(R["tab"]["ap"])
    print("\n  paired vs tab baseline:")
    for m in models[1:]:
        dau = np.array(R[m]["au"]) - base; dap = np.array(R[m]["ap"]) - base_ap
        print(f"    {m:20s} dAUROC {dau.mean():+.4f} (folds>0 {int((dau>0).sum())}/{K_FOLDS})   "
              f"dAUPRC {dap.mean():+.4f} (folds>0 {int((dap>0).sum())}/{K_FOLDS})")
    print(f"  (ref: plain tabular-DICE dice_gbdt AUROC 0.728 / AUPRC 0.251, SDoH)")

    # ---- stratification: GNN on all rows -> DICE K=4 ----
    print("\n=== stratification: DICE(K=4) on GNN embeddings (final fit) ===", flush=True)
    rng = np.random.default_rng(SEED); allr = np.arange(N).copy(); rng.shuffle(allr)
    nv = int(round(VAL_FRAC * N)); val, tr = allr[:nv], allr[nv:]
    E = gnn_embeddings(tr, val)
    fm = dice.fit(E, y, DK, DD, build_v(np.arange(N)))
    hard = dice.cluster_proba(fm, E).argmax(1)
    tiers = sorted(((int((hard == k).sum()),
                     float(y[hard == k].mean()) if (hard == k).any() else float("nan"), k)
                    for k in range(DK)), key=lambda t: (t[1] if t[1] == t[1] else 9))
    for n_k, r, k in tiers:
        za = anyz[hard == k].mean() * 100 if n_k else float("nan")
        print(f"  n={n_k:5d}  readmit={r*100:5.2f}%   SDOH_Any_Z in tier={za:4.1f}%")
    valid = [r for n_k, r, _ in tiers if r == r and r > 0 and n_k]
    if len(valid) >= 2:
        print(f"  high/low risk ratio = {max(valid)/min(valid):.2f}")
    r1, r0 = y[anyz == 1].mean(), y[anyz == 0].mean()
    print(f"  [anchor] SDOH_Any_Z flagged readmit {r1*100:.1f}% vs {r0*100:.1f}% (RR {r1/r0:.1f})")
    print("  (tabular-DICE tiers for comparison: 5.1 / 13.1 / 13.3 / 26.1%)")


if __name__ == "__main__":
    main()
