"""DICE-on-graph with RANDOM FOREST (tuned + moderate oversampling) as the
downstream learner, on the clean 41-feature SQL-fixed set. Tests whether RF
(the better tabular learner) improves DICE-on-graph and whether any variant
beats the tuned RF tabular ceiling (~0.718/0.187). Isotonic-calibrated pooled
OOF, matching analysis/table2_optimized.py. Reuses cv_dice_gnn.py's GNN
embedding + DICE machinery.

    PYTHONPATH=. python analysis/cv_dice_gnn_rf.py
"""
from __future__ import annotations
import os
os.environ.setdefault("DICE_SURROGATE_SIG", "1")           # bound DICE runtime (GNN dominates)

import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

import analysis.cv_dice_gnn as base          # reuses data load + gnn_embeddings/build_xtab/build_v/dice
import analysis.dice as dice

SEED, VAL_FRAC, CAL_CV = 42, 0.10, 3
SMOKE = os.environ.get("MEDHG_SMOKE") == "1"
K_FOLDS = 2 if SMOKE else 5
DK, DD = base.DK, base.DD
y, N = base.y, base.N

# tuned RF config from table2_optimized (reuse; sampling 'over' = RandomOverSampler 0.3)
RF_HP = dict(n_estimators=(120 if SMOKE else 200), max_depth=14, max_features="sqrt",
             min_samples_leaf=3, class_weight="balanced_subsample")
HGB_HP = dict(max_iter=200, learning_rate=0.037, max_leaf_nodes=40, l2_regularization=0.03)


def make_rf(sampling="over"):
    rf = RandomForestClassifier(n_estimators=RF_HP["n_estimators"], max_depth=RF_HP["max_depth"],
                                max_features=RF_HP["max_features"], min_samples_leaf=RF_HP["min_samples_leaf"],
                                class_weight=(None if sampling == "over" else RF_HP["class_weight"]),
                                random_state=SEED, n_jobs=-1)
    if sampling == "over":
        return ImbPipeline([("os", RandomOverSampler(sampling_strategy=0.3, random_state=SEED)), ("rf", rf)])
    return rf


def make_hgb():
    return HistGradientBoostingClassifier(random_state=SEED, **HGB_HP)


def make_lr():
    return Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=1000, class_weight="balanced"))])


def main():
    skf = StratifiedKFold(K_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.zeros(N), y))
    D = {}                                                  # per-fold design matrices (aligned to all N rows)
    rng = np.random.default_rng(SEED)
    for fi, (tr_all, te) in enumerate(folds):
        tr_s = tr_all.copy(); rng.shuffle(tr_s)
        nv = int(round(VAL_FRAC * N)); val, tr = tr_s[:nv], tr_s[nv:]
        full_tr = np.concatenate([tr, val])
        E = base.gnn_embeddings(tr, val)
        v = base.build_v(full_tr)
        dmodel = dice.fit(E[full_tr], y[full_tr], DK, DD, v[full_tr])
        chat = dice.cluster_proba(dmodel, E); z = dice.embed(dmodel, E); X = base.build_xtab(tr)
        D[fi] = dict(tab=X, enc=np.hstack([E, X]), chat=chat,
                     dice=np.hstack([chat, X]), full=np.hstack([z, chat, X]))
        print(f"[dgrf] fold {fi+1}/{K_FOLDS} designs built", flush=True)

    def cal_oof(make_est, key):
        p = np.full(N, np.nan)
        for fi, (tr, te) in enumerate(folds):
            X = D[fi][key]
            est = CalibratedClassifierCV(make_est(), method="isotonic", cv=CAL_CV)
            est.fit(X[tr], y[tr]); p[te] = est.predict_proba(X[te])[:, 1]
        return roc_auc_score(y, p), average_precision_score(y, p), brier_score_loss(y, p)

    specs = [
        ("tab (RF ceiling)",       make_rf,  "tab"),
        ("tab (HGB)",              make_hgb, "tab"),
        ("gnn_enc_stack (RF)",     make_rf,  "enc"),
        ("dice_gnn_only (RF)",     make_rf,  "chat"),
        ("dice_gnn (RF)",          make_rf,  "dice"),
        ("dice_gnn_full (RF)",     make_rf,  "full"),
        ("dice_gnn (HGB)",         make_hgb, "dice"),
        ("dice_gnn (LR)",          make_lr,  "dice"),
    ]
    res = {}
    for name, mk, key in specs:
        res[name] = cal_oof(mk, key)
        print(f"[dgrf] {name:22s} AUROC {res[name][0]:.3f}  AUPRC {res[name][1]:.3f}  Brier {res[name][2]:.4f}", flush=True)

    ceil = res["tab (RF ceiling)"]
    print(f"\n=== DICE-on-graph + RF (calibrated pooled OOF; N={N}, base {y.mean()*100:.2f}%) ===")
    print(f"  {'model':22s} {'AUROC':>7s} {'AUPRC':>7s} {'Brier':>7s}  dAUROC  dAUPRC (vs RF ceiling)")
    for name, _, _ in specs:
        au, ap, br = res[name]
        print(f"  {name:22s} {au:7.3f} {ap:7.3f} {br:7.4f}  {au-ceil[0]:+.3f}  {ap-ceil[1]:+.3f}")
    import csv
    with open("artifacts/newdata/cv_dice_gnn_rf.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "AUROC", "AUPRC", "Brier"])
        for name, _, _ in specs:
            w.writerow([name, *[f"{x:.4f}" for x in res[name]]])


if __name__ == "__main__":
    main()
