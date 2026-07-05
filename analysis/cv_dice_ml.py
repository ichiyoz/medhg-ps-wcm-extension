"""DICE -> ML pipeline for 30-day unplanned readmission ("DICE first, ML later").

Reproduces the PLOS pdig.0000606 two-stage design on the surgical cohort:
stage 1 runs DICE (Deep Significance Clustering; see analysis/dice.py) to
stratify patients into outcome-aware clusters and emit soft cluster-membership
probabilities AND a learned latent embedding; stage 2 feeds those into
downstream classifiers (memberships alone, memberships+tabular, embedding+
tabular, and a forward-feature-search LR -- the paper's DICE+LR+FFS recipe).

Fold-honest 5-fold CV on identical splits. Models:

  tab            : HGB on tabular + CPT (baseline)
  dice_only_lr   : LogisticRegression on DICE cluster memberships ALONE
  dice_lr        : LR  on [memberships + tabular]
  dice_gbdt      : HGB on [memberships + tabular]
  dice_lr_z      : LR  on [latent embedding z + tabular]
  dice_full_lr   : LR  on [z + memberships + tabular]
  dice_full_gbdt : HGB on [z + memberships + tabular]
  dice_lr_ffs    : LR  on forward-selected (<=20) cols of [memberships + z + tabular]
  dice_xgb       : XGBoost on [memberships + tabular]   (if xgboost installed)

DICE is fit on each fold's train rows only; (K, latent-d) chosen by a small NAS
on a carved-out validation split; memberships/embeddings re-extracted per fold
(no leakage). Age + sex are passed to the DICE outcome head as confounders.

    PYTHONPATH=. python analysis/cv_dice_ml.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import medhg_ps.config as C
from medhg_ps.data import apply_preprocess, fit_preprocess
from medhg_ps.deploy import assemble_training_frame
try:
    import analysis.dice as dice
except ModuleNotFoundError:
    import dice

SMOKE = os.environ.get("MEDHG_SMOKE") == "1"
K_FOLDS, SEED, VAL_FRAC = (2 if SMOKE else 5), 42, 0.10
FFS_CAP = 5 if SMOKE else 20

HGB = lambda: HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                             l2_regularization=1.0, random_state=42)
LR = lambda: LogisticRegression(max_iter=1000, class_weight="balanced")

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False


# ---- assemble the modelling frame (same path as the other cv_ scripts) -----
print(f"[dice] {'SMOKE ' if SMOKE else ''}loading...", flush=True)
merged, feat_cols, cpt_arr, Fseq, seq_all, y = assemble_training_frame()
y = np.asarray(y).astype(int)
N = len(merged)
print(f"[dice] cohort={N:,}  base={y.mean()*100:.2f}%  tabular feats={len(feat_cols)}  "
      f"exact_sig={dice.EXACT_SIG}  xgboost={'yes' if HAVE_XGB else 'no'}", flush=True)


def build_xtab(tr: np.ndarray) -> np.ndarray:
    """Standardized tabular + one-hot CPT design matrix (preprocessing fit on
    train rows only). Used as BOTH the DICE autoencoder input and the
    downstream tabular block."""
    _, st = fit_preprocess(merged[feat_cols].iloc[tr].reset_index(drop=True), id_cols=[])
    Xtab = apply_preprocess(merged[feat_cols], st)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(cpt_arr[tr])
    X = np.hstack([Xtab, ohe.transform(cpt_arr)])
    sc = StandardScaler().fit(X[tr])
    return sc.transform(X)


def build_v(fit_rows: np.ndarray) -> np.ndarray:
    """Confounders v for the DICE outcome head: standardized age + sex one-hot
    (scaler/encoder fit on fit_rows only)."""
    age = pd.to_numeric(merged["AgeYears"], errors="coerce").values.reshape(-1, 1)
    med = np.nanmedian(age[fit_rows])
    age = np.where(np.isnan(age), med, age)
    age = (age - age[fit_rows].mean()) / (age[fit_rows].std() + 1e-8)
    g = merged[["Gender"]].astype(str)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(g.iloc[fit_rows])
    return np.hstack([age, ohe.transform(g)]).astype(np.float32)


def _xgb():
    spw = float((y == 0).sum()) / max(float((y == 1).sum()), 1.0)
    return XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                         scale_pos_weight=spw, random_state=42, n_jobs=4)


def _pred_tree(model, D, full_tr, te):
    model.fit(D[full_tr], y[full_tr])
    return model.predict_proba(D[te])[:, 1]


def _pred_lr(D, full_tr, te):
    sc = StandardScaler().fit(D[full_tr])                  # LR wants scaled inputs
    m = LR().fit(sc.transform(D[full_tr]), y[full_tr])
    return m.predict_proba(sc.transform(D[te]))[:, 1]


def _ffs_select(Ctr, ytr, Cva, yva, cap):
    """Greedy forward feature selection by validation AUROC (paper's FFS)."""
    sel, rem, best = [], list(range(Ctr.shape[1])), -1.0
    while len(sel) < cap and rem:
        cand = None
        for j in rem:
            cols = sel + [j]
            m = LR().fit(Ctr[:, cols], ytr)
            a = roc_auc_score(yva, m.predict_proba(Cva[:, cols])[:, 1])
            if cand is None or a > cand[1]:
                cand = (j, a)
        if cand[1] <= best + 1e-4:
            break
        best = cand[1]; sel.append(cand[0]); rem.remove(cand[0])
    return sel


def main():
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    models = ["tab", "dice_only_lr", "dice_lr", "dice_gbdt", "dice_lr_z",
              "dice_full_lr", "dice_full_gbdt", "dice_lr_ffs"]
    if HAVE_XGB:
        models.append("dice_xgb")
    R = {m: {"au": [], "ap": []} for m in models}
    Ksel, dsel = [], []

    for fi, (tr_all, te) in enumerate(skf.split(np.zeros(N), y)):
        rng = np.random.default_rng(SEED + fi)
        tr_all = tr_all.copy(); rng.shuffle(tr_all)
        nv = int(round(VAL_FRAC * N))
        va, tr = tr_all[:nv], tr_all[nv:]                 # NAS/FFS split
        full_tr = np.concatenate([tr, va])

        X = build_xtab(tr)                                # design fit on inner-train rows
        v_nas = build_v(tr)                              # confounders for NAS
        v_fin = build_v(full_tr)                         # confounders for the refit

        # --- stage 1: DICE (NAS on tr->va, refit winner on full train) ---
        Kstar, dstar, vauc = dice.search_fit(X[tr], y[tr], X[va], y[va],
                                             v_nas[tr], v_nas[va])
        dmodel = dice.fit(X[full_tr], y[full_tr], Kstar, dstar, v_fin[full_tr], verbose=True)
        chat = dice.cluster_proba(dmodel, X)             # [N, K] soft membership
        z = dice.embed(dmodel, X)                        # [N, d] latent embedding
        Ksel.append(Kstar); dsel.append(dstar)

        # --- stage 2: downstream designs ---
        XD = np.hstack([chat, X])                        # memberships + tabular
        XZ = np.hstack([z, X])                           # embedding + tabular
        XF = np.hstack([z, chat, X])                     # embedding + memberships + tabular
        Cand = np.hstack([chat, z, X])                   # FFS candidate pool
        ff = _ffs_select(Cand[tr], y[tr], Cand[va], y[va], FFS_CAP)
        Cffs = Cand[:, ff]

        preds = {
            "tab":            _pred_tree(HGB(), X, full_tr, te),
            "dice_only_lr":   _pred_lr(chat, full_tr, te),
            "dice_lr":        _pred_lr(XD, full_tr, te),
            "dice_gbdt":      _pred_tree(HGB(), XD, full_tr, te),
            "dice_lr_z":      _pred_lr(XZ, full_tr, te),
            "dice_full_lr":   _pred_lr(XF, full_tr, te),
            "dice_full_gbdt": _pred_tree(HGB(), XF, full_tr, te),
            "dice_lr_ffs":    _pred_lr(Cffs, full_tr, te),
        }
        if HAVE_XGB:
            preds["dice_xgb"] = _pred_tree(_xgb(), XD, full_tr, te)

        for m in models:
            R[m]["au"].append(roc_auc_score(y[te], preds[m]))
            R[m]["ap"].append(average_precision_score(y[te], preds[m]))
        print(f"[dice] fold {fi+1}/{K_FOLDS}  K*={Kstar} d*={dstar} (val {vauc:.3f}) "
              f"ffs={len(ff)} | "
              + "  ".join(f"{m} {R[m]['au'][-1]:.3f}" for m in models), flush=True)

    # ---- results ----
    print(f"\n=== {K_FOLDS}-fold CV | base {y.mean()*100:.2f}% | K* {Ksel} d* {dsel} ===")
    print(f"  {'model':16s} {'AUROC':>16s} {'AUPRC':>16s}")
    for m in models:
        au, ap = np.array(R[m]["au"]), np.array(R[m]["ap"])
        print(f"  {m:16s} {au.mean():.3f} +/- {au.std():.3f}    {ap.mean():.3f} +/- {ap.std():.3f}")

    base, base_ap = np.array(R["tab"]["au"]), np.array(R["tab"]["ap"])
    prev = 0.719                                          # previous best (dice_gbdt)
    print("\n  paired vs tab baseline:")
    for m in models:
        if m == "tab":
            continue
        dau = np.array(R[m]["au"]) - base
        dap = np.array(R[m]["ap"]) - base_ap
        print(f"    {m:16s} dAUROC {dau.mean():+.4f} (folds>0 {int((dau>0).sum())}/{K_FOLDS})   "
              f"dAUPRC {dap.mean():+.4f} (folds>0 {int((dap>0).sum())}/{K_FOLDS})")
    print(f"\n  (reference: previous DICE best dice_gbdt AUROC {prev})")

    # ---- DICE cluster risk stratification (final fit on all rows) ----
    # Locked to K=4: low / medium / high + a small SDoH-insecurity very-high tier.
    # The very-high tier is seed-sensitive (the SDoH-flagged group is ~2.6% of the
    # cohort); the robust anchor is the raw SDOH_Any_Z association printed below.
    Kfinal, dfinal = 4, 16
    Xall, vall = build_xtab(np.arange(N)), build_v(np.arange(N))
    fmodel = dice.fit(Xall, y, Kfinal, dfinal, vall, verbose=True)
    hard = dice.cluster_proba(fmodel, Xall).argmax(1)
    anyz = pd.to_numeric(merged.get("SDOH_Any_Z", 0), errors="coerce").fillna(0).values
    print(f"\n=== DICE cluster risk stratification (final fit, K={Kfinal} d={dfinal}) ===")
    tiers = sorted(((int((hard == k).sum()),
                     float(y[hard == k].mean()) if (hard == k).any() else float("nan"), k)
                    for k in range(Kfinal)), key=lambda t: (t[1] if t[1] == t[1] else 9))
    for n_k, r, k in tiers:
        za = anyz[hard == k].mean() * 100 if n_k else float("nan")
        print(f"  n={n_k:5d}  readmit={r*100:5.2f}%   SDOH_Any_Z in tier={za:4.1f}%")
    valid = [r for n_k, r, _ in tiers if r == r and r > 0 and n_k]
    if len(valid) >= 2:
        print(f"  high/low risk ratio = {max(valid)/min(valid):.2f}")
    if anyz.sum():                                          # seed-independent anchor
        r1, r0 = y[anyz == 1].mean(), y[anyz == 0].mean()
        print(f"  [anchor] SDOH_Any_Z flagged n={int(anyz.sum())} ({anyz.mean()*100:.1f}%): "
              f"readmit {r1*100:.1f}% vs {r0*100:.1f}% unflagged (RR {r1/r0:.1f})")


if __name__ == "__main__":
    main()
