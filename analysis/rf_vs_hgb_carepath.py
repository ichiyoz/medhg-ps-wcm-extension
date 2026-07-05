"""RF vs HGB, each with/without the GRU care-path encoder, SQL-fixed data.
One protocol for all cells: 5-fold CV, GRU trained per fold on train rows only,
isotonic-calibrated classifiers, pooled out-of-fold AUROC/AUPRC/Brier (matches
the Table-2 decision-curve protocol, so AUPRC is comparable to gbt_gru 0.187).
Reuses analysis/cv_seq_gru.py's SeqGRU/train_gru/design and its data assembly.
"""
import os
os.environ.setdefault("MEDHG_PS_DEVICE", "cpu")
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold

import analysis.cv_seq_gru as G   # importing runs its data assembly

y, N = G.y_all, G.N
K, SEED, VAL_FRAC = 5, 42, 0.10
RF = lambda: RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                    random_state=42, n_jobs=-1)
HGB = lambda: HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                             l2_regularization=1.0, random_state=42)

cells = ["rf_tab", "hgb_tab", "rf_tab_gru", "hgb_tab_gru"]
oof = {c: np.full(N, np.nan) for c in cells}
skf = StratifiedKFold(K, shuffle=True, random_state=SEED)
print(f"[rfhgb] N={N} base={y.mean():.4f} feats={len(G.feat_cols)}", flush=True)
for fi, (tr_all, te) in enumerate(skf.split(np.zeros(N), y)):
    rng = np.random.default_rng(SEED + fi); tr_all = tr_all.copy(); rng.shuffle(tr_all)
    nv = int(round(VAL_FRAC * N)); va, tr = tr_all[:nv], tr_all[nv:]
    emb, praw, vb = G.train_gru(tr, va)
    Xtab = G.design(tr, False, False, emb)          # clinical + CPT
    Xtabg = G.design(tr, False, True, emb)          # + GRU care-path embedding
    for c, X, clf in [("rf_tab", Xtab, RF), ("hgb_tab", Xtab, HGB),
                      ("rf_tab_gru", Xtabg, RF), ("hgb_tab_gru", Xtabg, HGB)]:
        est = CalibratedClassifierCV(clf(), method="isotonic", cv=3).fit(X[tr_all], y[tr_all])
        oof[c][te] = est.predict_proba(X[te])[:, 1]
    print(f"[rfhgb] fold {fi+1}/{K} done (gru val {vb:.3f})", flush=True)

print(f"\n=== RF vs HGB, +/- GRU care-path | isotonic-calibrated pooled OOF | N={N} base={y.mean()*100:.2f}% ===")
print(f"  {'model':14s} {'AUROC':>7} {'AUPRC':>7} {'Brier':>7}")
for c in cells:
    p = oof[c]
    print(f"  {c:14s} {roc_auc_score(y,p):7.3f} {average_precision_score(y,p):7.3f} {brier_score_loss(y,p):7.4f}", flush=True)
print(f"\n  (reference: Table-2 gbt_gru AUPRC 0.187; tabular RF cv_variant3 0.715/0.180)")
