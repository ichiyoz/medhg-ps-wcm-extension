"""Variable-selection and class-imbalance-sampling experiments on the tabular
readmission model (SQL-fixed data). 5-fold, pooled out-of-fold AUROC/AUPRC.
All selection/resampling fit on the training fold only. CPU."""
import os, warnings, time
os.environ.setdefault("MEDHG_PS_DEVICE", "cpu")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from medhg_ps.data import fit_preprocess, apply_preprocess
from medhg_ps.deploy import assemble_training_frame

t0 = time.time()
merged, feat_cols, cpt_arr, Fseq, seq_all, y = assemble_training_frame()
y = np.asarray(y).astype(int); N = len(merged)
print(f"N={N} base={y.mean():.4f} tabular feats={len(feat_cols)}", flush=True)

def design(tr):
    _, st = fit_preprocess(merged[feat_cols].iloc[tr].reset_index(drop=True), id_cols=[])
    Xtab = apply_preprocess(merged[feat_cols], st)
    oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(cpt_arr[tr])
    return np.hstack([Xtab, oh.transform(cpt_arr)])

skf = StratifiedKFold(5, shuffle=True, random_state=42)
folds = [(design(tr), tr, te) for tr, te in skf.split(np.zeros(N), y)]
print(f"design built ({folds[0][0].shape[1]} cols) t={time.time()-t0:.0f}s", flush=True)

def sw(yt):
    w = np.ones(len(yt)); n1 = yt.sum(); n0 = len(yt) - n1
    w[yt == 1] = len(yt) / (2 * n1); w[yt == 0] = len(yt) / (2 * n0); return w
def HGB(): return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, l2_regularization=1.0, random_state=42)
def RF():  return RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=4)

def pooled(fn):
    p = np.full(N, np.nan)
    for X, tr, te in folds:
        p[te] = fn(X, tr, te)
    return roc_auc_score(y, p), average_precision_score(y, p), brier_score_loss(y, p)

# ---------- Experiment A: variable selection (HGB) ----------
def fit_hgb(Xtr, ytr, Xte, cols=None):
    if cols is not None:
        Xtr, Xte = Xtr[:, cols], Xte[:, cols]
    m = HGB(); m.fit(Xtr, ytr, sample_weight=sw(ytr)); return m.predict_proba(Xte)[:, 1]

def m_full(X, tr, te): return fit_hgb(X[tr], y[tr], X[te])
def m_lasso(X, tr, te):
    sc = StandardScaler().fit(X[tr]); Xs = sc.transform(X[tr])
    l = LogisticRegression(penalty="l1", solver="liblinear", C=0.1, class_weight="balanced", max_iter=500).fit(Xs, y[tr])
    cols = np.where(l.coef_[0] != 0)[0]
    if len(cols) == 0: cols = np.arange(X.shape[1])
    m_lasso.nfeat = len(cols)
    return fit_hgb(X[tr], y[tr], X[te], cols)
def make_topk(K):
    def f(X, tr, te):
        imp = RandomForestClassifier(100, class_weight="balanced", random_state=42, n_jobs=4).fit(X[tr], y[tr]).feature_importances_
        cols = np.argsort(imp)[::-1][:K]
        return fit_hgb(X[tr], y[tr], X[te], cols)
    return f
def m_rfe(X, tr, te):
    sc = StandardScaler().fit(X[tr])
    r = RFE(LogisticRegression(class_weight="balanced", max_iter=400), n_features_to_select=20, step=0.15).fit(sc.transform(X[tr]), y[tr])
    cols = np.where(r.support_)[0]
    return fit_hgb(X[tr], y[tr], X[te], cols)
def m_ffs(X, tr, te):
    xi, xv, yi, yv = train_test_split(X[tr], y[tr], test_size=0.25, stratify=y[tr], random_state=42)
    sel, rem, best = [], list(range(X.shape[1])), -1.0
    while len(sel) < 20 and rem:
        cand = None
        for j in rem:
            cols = sel + [j]
            mm = LogisticRegression(class_weight="balanced", max_iter=300).fit(xi[:, cols], yi)
            a = roc_auc_score(yv, mm.predict_proba(xv[:, cols])[:, 1])
            if cand is None or a > cand[1]: cand = (j, a)
        if cand[1] <= best + 1e-4: break
        best = cand[1]; sel.append(cand[0]); rem.remove(cand[0])
    m_ffs.nfeat = len(sel)
    return fit_hgb(X[tr], y[tr], X[te], sel)

print("\n=== Experiment A: variable selection (HGB) ===", flush=True)
print(f"{'method':16s} {'nfeat':>6} {'AUROC':>7} {'AUPRC':>7}", flush=True)
A = [("full", m_full, folds[0][0].shape[1]), ("lasso", m_lasso, None),
     ("top10", make_topk(10), 10), ("top20", make_topk(20), 20), ("top30", make_topk(30), 30),
     ("rfe20", m_rfe, 20), ("ffs<=20", m_ffs, None)]
for name, fn, nf in A:
    au, ap, _ = pooled(fn)
    nfd = getattr(fn, "nfeat", nf)
    print(f"{name:16s} {str(nfd):>6} {au:7.3f} {ap:7.3f}  t={time.time()-t0:.0f}s", flush=True)

# ---------- Experiment B: sampling (HGB and RF, full features) ----------
try:
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    HAVE_IMB = True
except Exception:
    HAVE_IMB = False
    print("\n[imbalanced-learn unavailable — SMOTE/random samplers skipped unless installed]", flush=True)

def clf(kind): return HGB() if kind == "hgb" else RandomForestClassifier(300, random_state=42, n_jobs=4)
def run_sample(kind, sampler):
    def fn(X, tr, te):
        Xtr, ytr = X[tr], y[tr]
        if sampler == "cw":                                  # class-weighted baseline (no resample)
            m = HGB() if kind == "hgb" else RF()
            if kind == "hgb": m.fit(Xtr, ytr, sample_weight=sw(ytr))
            else: m.fit(Xtr, ytr)
            return m.predict_proba(X[te])[:, 1]
        Xr, yr = sampler.fit_resample(Xtr, ytr)              # resample train only
        m = clf(kind); m.fit(Xr, yr); return m.predict_proba(X[te])[:, 1]
    return fn

print("\n=== Experiment B: sampling ===", flush=True)
print(f"{'learner':6s} {'method':14s} {'AUROC':>7} {'AUPRC':>7} {'Brier':>7}", flush=True)
methods = [("class-weight", "cw")]
if HAVE_IMB:
    methods += [("oversample", RandomOverSampler(random_state=42)),
                ("undersample", RandomUnderSampler(random_state=42)),
                ("SMOTE", SMOTE(random_state=42)),
                ("over-0.3", RandomOverSampler(sampling_strategy=0.3, random_state=42))]
for kind in ["hgb", "rf"]:
    for name, s in methods:
        au, ap, br = pooled(run_sample(kind, s))
        print(f"{kind:6s} {name:14s} {au:7.3f} {ap:7.3f} {br:7.4f}  t={time.time()-t0:.0f}s", flush=True)
print("DONE", flush=True)
