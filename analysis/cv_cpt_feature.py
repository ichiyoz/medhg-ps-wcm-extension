"""Does CPT help as a FEATURE? tabular vs tabular+CPT (5-fold CV).

The CPT hypergraph only tied tabular, suggesting CPT carries signal the tabular
model lacks (it has procedure COUNTS, not the code). The right way to use an
informative variable is as a feature. This adds PrimaryCPT (46 codes,
one-hot) as a categorical and CV-compares tabular vs tabular+CPT with RF and
HGB, reporting the paired per-fold lift. Self-loads CPT from the cohort CSV.

    PYTHONPATH=. python analysis/cv_cpt_feature.py
"""
import glob
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import medhg_ps.config as C
from medhg_ps.data import (load_raw, build_preop_trajectory_features, add_calendar_features)

K, SEED = 5, 42
COHORT_COLS = ["LogID", "EncounterCSN", "PAT_ID", "SurgeryDate", "SurgeryYear", "PrimaryCPT", "AgeYears"]


def _norm(s):
    return s.astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)


def load_cpt():
    for d in [str(C.DATA_DIR), "/Users/yiyezhang/Dropbox/Surgery"]:
        for p in glob.glob(os.path.join(d, "*.csv")):
            try:
                head = pd.read_csv(p, nrows=1, header=None, dtype=str)
            except Exception:
                continue
            if str(head.iloc[0, 0]).strip() == "LogID":
                df = pd.read_csv(p, low_memory=False)
                if {"LogID", "PrimaryCPT"} <= set(df.columns):
                    return dict(zip(_norm(df["LogID"]), _norm(df["PrimaryCPT"])))
            elif head.shape[1] == len(COHORT_COLS):
                df = pd.read_csv(p, header=None, names=COHORT_COLS, low_memory=False)
                return dict(zip(_norm(df["LogID"]), _norm(df["PrimaryCPT"])))
    raise FileNotFoundError("no cohort CSV with PrimaryCPT found")


cpt_map = load_cpt()
raw = load_raw()
enc_nodupes = raw.enc_features.drop(
    columns=([c for c in raw.encounters.columns
              if c != "LogID" and c in raw.enc_features.columns]
             + ["ReadmittedWithin30Days"]), errors="ignore")
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
merged["PrimaryCPT"] = [cpt_map.get(l, "UNK") for l in merged["LogID"]]
feat_cols = ([c for c in C.MODEL_FEATURE_COLUMNS if c in merged.columns]
             + [c for c in C.ENCOUNTER_DERIVED_COLUMNS if c in merged.columns])
y = merged["ReadmittedWithin30Days"].astype(int).values
N = len(merged)
print(f"cohort={N:,}  base={y.mean()*100:.2f}%  tabular feats={len(feat_cols)}  "
      f"CPT codes={merged['PrimaryCPT'].nunique()}\n")


def _pre(X):
    cat = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    return ColumnTransformer([
        ("c", Pipeline([("i", SimpleImputer(strategy="constant", fill_value="Unknown")),
                        ("o", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
        ("n", Pipeline([("i", SimpleImputer(strategy="mean")), ("s", StandardScaler())]), num)])


def fit_eval(est_fn, cols, tr, te):
    X = merged[cols]
    pipe = Pipeline([("p", _pre(X)), ("c", est_fn())])
    pipe.fit(X.iloc[tr], y[tr])
    p = pipe.predict_proba(X.iloc[te])[:, 1]
    return roc_auc_score(y[te], p), average_precision_score(y[te], p)


def rf():
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=10, max_features="sqrt",
                                  class_weight="balanced", random_state=42, n_jobs=-1)


def hgb():
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                          l2_regularization=1.0, random_state=42)


skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
res = {(m, f): {"au": [], "ap": []} for m in ("RF", "HGB") for f in ("tab", "tab+CPT")}
for tr, te in skf.split(np.zeros(N), y):
    for mname, fn in (("RF", rf), ("HGB", hgb)):
        a0, p0 = fit_eval(fn, feat_cols, tr, te)
        a1, p1 = fit_eval(fn, feat_cols + ["PrimaryCPT"], tr, te)
        res[(mname, "tab")]["au"].append(a0); res[(mname, "tab")]["ap"].append(p0)
        res[(mname, "tab+CPT")]["au"].append(a1); res[(mname, "tab+CPT")]["ap"].append(p1)

print(f"=== {K}-fold CV (mean +/- std) | base {y.mean()*100:.2f}% ===")
print(f"  {'model':14s} {'AUROC':>16s} {'AUPRC':>16s}")
for key in [("RF", "tab"), ("RF", "tab+CPT"), ("HGB", "tab"), ("HGB", "tab+CPT")]:
    au, ap = np.array(res[key]["au"]), np.array(res[key]["ap"])
    print(f"  {key[0]+' '+key[1]:14s} {au.mean():.3f} +/- {au.std():.3f}    {ap.mean():.3f} +/- {ap.std():.3f}")

print("\n=== Paired lift  (tab+CPT) - tab ===")
for m in ("RF", "HGB"):
    da = np.array(res[(m, "tab+CPT")]["au"]) - np.array(res[(m, "tab")]["au"])
    dp = np.array(res[(m, "tab+CPT")]["ap"]) - np.array(res[(m, "tab")]["ap"])
    print(f"  {m}: dAUROC {da.mean():+.4f} (folds>0 {int((da>0).sum())}/{K})  {np.round(da,3).tolist()} | "
          f"dAUPRC {dp.mean():+.4f} (folds>0 {int((dp>0).sum())}/{K})  {np.round(dp,3).tolist()}")
