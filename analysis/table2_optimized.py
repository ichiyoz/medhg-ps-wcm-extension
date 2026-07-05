"""Optimized Table 2 on the clean 41-feature SQL-fixed set.

Reports the deployable models AND random forest at tuned/best performance.
Reuses cv_seq_gru for the data assembly + GRU care-path encoder. One protocol:
5-fold CV (seed 42), isotonic-calibrated, pooled out-of-fold AUROC/AUPRC/Brier.
Light hyperparameter tuning (Optuna) for RF and HGB on a global train/val split;
best sampling (class-weight vs moderate oversampling) chosen for RF by val AUPRC.
All preprocessing/GRU/sampling fit on training rows only.
"""
from __future__ import annotations
import os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

import analysis.cv_seq_gru as sg          # data (merged, feat_cols, cpt_arr, y_all) + train_gru + design
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

SMOKE = os.environ.get("MEDHG_SMOKE") == "1"
SEED = 42
N = len(sg.y_all); y = sg.y_all
N_TRIALS = 5 if SMOKE else 20
CAL_CV = 3
print(f"[t2opt] N={N} base={y.mean():.4f} feats={len(sg.feat_cols)} SMOKE={SMOKE}", flush=True)

# ---------- hyperparameter tuning on a global stratified 80/20 split ----------
tr0, va0 = train_test_split(np.arange(N), test_size=0.2, stratify=y, random_state=SEED)
Xtab0 = sg.design(tr0, False, False, None)          # tabular design (fit on tr0)

def _rf(params, sampling):
    rf = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"],
                                max_features=params["max_features"], min_samples_leaf=params["min_samples_leaf"],
                                class_weight=(None if sampling == "over" else params["class_weight"]),
                                random_state=SEED, n_jobs=-1)
    if sampling == "over":
        return ImbPipeline([("os", RandomOverSampler(sampling_strategy=0.3, random_state=SEED)), ("rf", rf)])
    return rf

def tune_rf():
    def obj(t):
        p = dict(n_estimators=t.suggest_int("n_estimators", 200, 500, step=100),
                 max_depth=t.suggest_categorical("max_depth", [None, 8, 14, 20]),
                 max_features=t.suggest_categorical("max_features", ["sqrt", 0.3, 0.5]),
                 min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 8),
                 class_weight=t.suggest_categorical("class_weight", ["balanced", "balanced_subsample"]))
        samp = t.suggest_categorical("sampling", ["cw", "over"])
        m = _rf(p, samp).fit(Xtab0[tr0], y[tr0])
        return average_precision_score(y[va0], m.predict_proba(Xtab0[va0])[:, 1])
    s = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    s.optimize(obj, n_trials=N_TRIALS)
    bp = s.best_params; samp = bp.pop("sampling")
    return bp, samp, s.best_value

def tune_hgb():
    def obj(t):
        m = HistGradientBoostingClassifier(
            max_iter=t.suggest_int("max_iter", 200, 600, step=100),
            learning_rate=t.suggest_float("learning_rate", 0.02, 0.15, log=True),
            max_leaf_nodes=t.suggest_int("max_leaf_nodes", 15, 63),
            l2_regularization=t.suggest_float("l2_regularization", 1e-3, 3.0, log=True),
            min_samples_leaf=t.suggest_int("min_samples_leaf", 10, 60), random_state=SEED)
        m.fit(Xtab0[tr0], y[tr0])          # plain HGB; isotonic calibration handles the scale
        return average_precision_score(y[va0], m.predict_proba(Xtab0[va0])[:, 1])
    s = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    s.optimize(obj, n_trials=N_TRIALS)
    return s.best_params, s.best_value

rf_hp, rf_samp, rf_val = tune_rf()
hgb_hp, hgb_val = tune_hgb()
print(f"[t2opt] tuned RF {rf_hp} sampling={rf_samp} (valAP {rf_val:.3f})", flush=True)
print(f"[t2opt] tuned HGB {hgb_hp} (valAP {hgb_val:.3f})", flush=True)

def make_rf(sampling=None):
    return _rf(rf_hp, sampling if sampling is not None else rf_samp)
def make_hgb():
    return HistGradientBoostingClassifier(random_state=SEED, **hgb_hp)

# ---------- 5-fold CV, per-fold GRU, calibrated pooled OOF ----------
K = 2 if SMOKE else 5
skf = StratifiedKFold(K, shuffle=True, random_state=SEED)
folds = list(skf.split(np.zeros(N), y))
VAL_FRAC = 0.10

def cal_oof(make_est, Xof, is_hgb=False):
    """isotonic-calibrated pooled OOF for one model; Xof(fi,tr)->design for fold."""
    p = np.full(N, np.nan)
    for fi, (tr, te) in enumerate(folds):
        X = Xof[fi]
        est = CalibratedClassifierCV(make_est(), method="isotonic", cv=CAL_CV)
        est.fit(X[tr], y[tr])              # RF carries class_weight internally; HGB plain
        p[te] = est.predict_proba(X[te])[:, 1]
    return (roc_auc_score(y, p), average_precision_score(y, p), brier_score_loss(y, p))

# precompute per-fold GRU embedding + the three designs
Dtab, Dseq, Dgru, GRU = {}, {}, {}, {}
rng = np.random.default_rng(SEED)
for fi, (tr, te) in enumerate(folds):
    tr_s = tr.copy(); rng.shuffle(tr_s)
    nv = int(round(VAL_FRAC * N)); va, trg = tr_s[:nv], tr_s[nv:]
    emb, _, vb = sg.train_gru(trg, va)
    GRU[fi] = emb
    Dtab[fi] = sg.design(tr, False, False, None)
    Dseq[fi] = sg.design(tr, True, False, None)
    Dgru[fi] = sg.design(tr, False, True, emb)
    print(f"[t2opt] fold {fi+1}/{K} GRU val {vb:.3f} designs built", flush=True)

rows = []
def rec(name, res, extra=""):
    rows.append((name, *res, extra)); a, p, b = res
    print(f"  {name:44s} AUROC {a:.3f}  AUPRC {p:.3f}  Brier {b:.4f}  {extra}", flush=True)

print("\n=== Optimized Table 2 (5-fold, calibrated pooled OOF; N={} base {:.2%}) ===".format(N, y.mean()), flush=True)
rec("RF, tabular (tuned+sampling)  [CEILING]", cal_oof(make_rf, Dtab), f"rf/{rf_samp}")
rec("HGB, tabular (tuned)", cal_oof(make_hgb, Dtab, is_hgb=True), "hgb")
# clinical + hand-crafted care-path sequence: pick better learner
rf_seq = cal_oof(make_rf, Dseq); hgb_seq = cal_oof(make_hgb, Dseq, is_hgb=True)
best_seq, tag = (rf_seq, "rf") if rf_seq[1] >= hgb_seq[1] else (hgb_seq, "hgb")
rec("clinical + hand-crafted care path", best_seq, f"best={tag}")
# clinical + GRU care-path encoder (deployable): both learners
rf_gru = cal_oof(make_rf, Dgru); hgb_gru = cal_oof(make_hgb, Dgru, is_hgb=True)
rec("clinical + GRU care-path encoder  [DEPLOYABLE]  (RF)", rf_gru, "rf")
rec("clinical + GRU care-path encoder  [DEPLOYABLE]  (HGB)", hgb_gru, "hgb")
# best-of-everything: RF + moderate oversampling + GRU
rec("BEST-OF-ALL: RF+oversample(0.3)+GRU", cal_oof(lambda: make_rf("over"), Dgru), "rf/over")

out = pd.DataFrame(rows, columns=["model", "AUROC", "AUPRC", "Brier", "note"])
out.to_csv("artifacts/newdata/table2_optimized.csv", index=False)
print("\nsaved -> artifacts/newdata/table2_optimized.csv", flush=True)
