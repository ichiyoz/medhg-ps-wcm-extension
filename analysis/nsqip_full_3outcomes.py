"""Tabular RF on the FULL NSQIP cohort (N=6,014) for three outcomes side-by-side.

Outcomes (all binary):
  1. mortality        = Postop Death w/in 30 days == "Yes"   (28 events, 0.47%)
  2. still_in_hospital = Still in Hospital >30 Days == "Yes" (43 events, 0.72%)
  3. readmission30    = # of Unplanned Readmissions > 0      (309 events, 5.14%; reference)

Protocol (identical across outcomes):
  - 5-fold StratifiedKFold seed 42
  - Isotonic-calibrated pooled OOF (CalibratedClassifierCV method='isotonic', cv=3)
  - RF canonical: 500 trees, min_leaf 10, sqrt, class_weight='balanced'
  - Bootstrap n=2000 CIs

Model variants:
  A. rf_full        — RF on all NSQIP features (baseline)
  B. rf_no_lab      — RF with 12 lab columns dropped (sensitivity to lab missingness)
  C. rf_oversample  — RF with class_weight=None + RandomOverSampler(0.3) on train fold
"""
from __future__ import annotations
import re, json, numpy as np, pandas as pd, warnings
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import RandomOverSampler

import medhg_ps.config as C
from medhg_ps.data import fit_preprocess, apply_preprocess
from medhg_ps.evaluate import _bootstrap_ci

warnings.filterwarnings("ignore")
SEED = 42
K_FOLDS = 5
EXT = "/Users/yiyezhang/Downloads/Case_Details_and_Custom_Fields_Report-01-Apr-2025-0916.xlsx"
OUT_DIR = Path("/Users/yiyezhang/Library/CloudStorage/Dropbox/Surgery/medhg-ps/artifacts/newdata")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 1. Load + harmonize NSQIP features (same recipe as medhgps_nsqip_matched.py) ----------
print("[t] loading NSQIP xlsx...", flush=True)
ext = pd.read_excel(EXT)
print(f"[t] full NSQIP N={len(ext):,}", flush=True)

wunit = ext.get("Weight Unit", pd.Series(["kg"] * len(ext))).astype(str).str.lower()
ext = ext.rename(columns=C.NSQIP_TO_SQL_RENAMES)
ext["Gender"] = ext["Gender"].map({"Female": "F", "Male": "M"}).fillna(ext["Gender"])
ext["PatientType"] = ext["PatientType"].map({"Inpatient": "I", "Outpatient": "O"}).fillna(ext["PatientType"])
_roman = {"I":"1.0","II":"2.0","III":"3.0","IV":"4.0","V":"5.0","VI":"6.0"}
ext["ASAClass"] = ext["ASAClass"].map(
    lambda s:(lambda m: _roman.get(m.group(1)) if m else np.nan)(re.search(r"ASA\s+(VI|IV|V|III|II|I)\b", str(s)))
)
_anes = {"General":"general","Regional":"regional","Spinal":"spinal","Epidural":"epidural",
         "Monitored anesthesia care/IV sedation":"MAC"}
ext["AnesType"] = ext["AnesType"].map(_anes).fillna(ext["AnesType"].astype(str).str.lower())

def _disp(s):
    s = str(s).lower()
    if "ama" in s or "against medical" in s: return "AMA"
    if "expired" in s: return "Expired"
    if "home" in s or "self care" in s or "hospice/home" in s: return "Home"
    if any(k in s for k in ["facility","hospital","nursing","rehab","custodial","skilled"]): return "Facility"
    return "Other"
ext["Discharge Disposition"] = ext["Discharge Disposition"].map(_disp)

w = pd.to_numeric(ext["Weight (kg)"], errors="coerce")
ext["Weight (kg)"] = np.where(wunit.str.startswith("lb"), w * 0.453592, w)
ext["PrimaryCPT"] = ext.get("CPT Code", pd.Series([np.nan]*len(ext))).astype(str).str.replace(r"\.0+$","",regex=True)

# ---------- 2. Feature set — the columns from NSQIP_TO_SQL_RENAMES + PrimaryCPT ----------
SDOH = {"Language","SDOH_Housing_Z","SDOH_Food_Z","SDOH_Financial_Z","SDOH_Any_Z"}
FEATS = [f for f in C.MODEL_FEATURE_COLUMNS
         if f not in SDOH and f in ext.columns and f != "PrimaryCPT"]
LAB_COLS = ["NA","BUN","Creat","ALB","BT","SGOT","ALKPhos","WBC","HCT","PLT","INR","APTT"]
FEATS_TAB = FEATS[:]
FEATS_NO_LAB = [f for f in FEATS if f not in LAB_COLS]
print(f"[t] full feature set (n={len(FEATS_TAB)}): {FEATS_TAB}", flush=True)
print(f"[t] no-lab feature set (n={len(FEATS_NO_LAB)}): {FEATS_NO_LAB}", flush=True)

# ---------- 3. Outcomes ----------
def _bool_yes(col):
    return (ext[col].astype(str).str.strip().str.lower() == "yes").astype(int).values

y_mort = _bool_yes("Postop Death w/in 30 days of Procedure")
y_sih  = _bool_yes("Still in Hospital >30 Days")
y_read = (pd.to_numeric(ext.get("# of Unplanned Readmissions", 0), errors="coerce").fillna(0) > 0).astype(int).values

OUTCOMES = [
    ("mortality",        y_mort, 0.005),
    ("still_in_hospital", y_sih, 0.007),
    ("readmission30",    y_read, 0.051),
]

def _rf():
    return RandomForestClassifier(
        n_estimators=500, min_samples_leaf=10, max_features="sqrt",
        class_weight="balanced", random_state=SEED, n_jobs=-1
    )

def _rf_unweighted():
    return RandomForestClassifier(
        n_estimators=500, min_samples_leaf=10, max_features="sqrt",
        class_weight=None, random_state=SEED, n_jobs=-1
    )

def _cv_oof(feat_cols, y, model_kind):
    """Return calibrated pooled-OOF probabilities and per-fold event counts."""
    N = len(y); p = np.full(N, np.nan)
    skf = StratifiedKFold(K_FOLDS, shuffle=True, random_state=SEED)
    per_fold_events = []
    for fi, (tr, te) in enumerate(skf.split(np.zeros(N), y)):
        # design
        _, st = fit_preprocess(ext[feat_cols].iloc[tr].reset_index(drop=True), id_cols=[])
        X = apply_preprocess(ext[feat_cols], st)
        # one-hot CPT
        cpt = np.asarray(ext["PrimaryCPT"].astype(str)).reshape(-1, 1)
        oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(cpt[tr])
        X = np.hstack([X, oh.transform(cpt)])
        n_events_te = int(y[te].sum()); per_fold_events.append(n_events_te)
        # train
        try:
            if model_kind == "oversample":
                base = _rf_unweighted()
                ros = RandomOverSampler(sampling_strategy=0.3, random_state=SEED)
                Xtr, ytr = ros.fit_resample(X[tr], y[tr])
                est = CalibratedClassifierCV(base, method="isotonic", cv=3).fit(Xtr, ytr)
            else:
                base = _rf()
                est = CalibratedClassifierCV(base, method="isotonic", cv=3).fit(X[tr], y[tr])
            p[te] = est.predict_proba(X[te])[:, 1]
        except Exception as e:
            print(f"    [warn] fold {fi} failed: {e}", flush=True)
            p[te] = y.mean()  # baseline fallback
    return p, per_fold_events

def _report(name, y, p):
    au = roc_auc_score(y, p)
    ap = average_precision_score(y, p)
    br = brier_score_loss(y, p)
    au_ci = _bootstrap_ci(y, p, roc_auc_score, n_boot=2000, seed=0)
    ap_ci = _bootstrap_ci(y, p, average_precision_score, n_boot=2000, seed=1)
    return {"model": name, "AUROC": au, "AUROC_lo": au_ci[0], "AUROC_hi": au_ci[1],
            "AUPRC": ap, "AUPRC_lo": ap_ci[0], "AUPRC_hi": ap_ci[1], "Brier": br}

def _perm_importance_top5(feat_cols, y, p, model_kind):
    """Recompute a single full-data RF to get permutation importance top-5.
    Aggregates one-hot expansions back to original feature names."""
    _, st = fit_preprocess(ext[feat_cols], id_cols=[])
    X = apply_preprocess(ext[feat_cols], st)
    cpt = np.asarray(ext["PrimaryCPT"].astype(str)).reshape(-1, 1)
    oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(cpt)
    X = np.hstack([X, oh.transform(cpt)])
    if model_kind == "oversample":
        ros = RandomOverSampler(sampling_strategy=0.3, random_state=SEED)
        Xtr, ytr = ros.fit_resample(X, y)
        rf = _rf_unweighted().fit(Xtr, ytr)
    else:
        rf = _rf().fit(X, y)
    tab_cols = list(st.final_feature_names)
    cpt_cols = [f"PrimaryCPT={c}" for c in oh.categories_[0]]
    all_cols = tab_cols + cpt_cols
    cat_set = set(st.categorical_cols)
    def _orig(col):
        if col.startswith("PrimaryCPT="): return "PrimaryCPT"
        for c in cat_set:
            if col.startswith(c + "_") or col == c: return c
        return col
    pi = permutation_importance(rf, X, y, n_repeats=3, random_state=SEED,
                                n_jobs=-1, scoring="average_precision")
    perm = pd.Series(pi.importances_mean, index=all_cols)
    agg = perm.groupby([_orig(c) for c in perm.index]).sum().sort_values(ascending=False)
    return agg.head(5)

# ---------- 4. Loop over outcomes × models ----------
rows = []
outcome_summary = {}
for name, y, no_skill_ref in OUTCOMES:
    n_events = int(y.sum())
    print(f"\n{'='*60}", flush=True)
    print(f"[t] OUTCOME = {name}: n_events={n_events} / {len(y)} ({n_events/len(y)*100:.2f}%)  "
          f"no-skill AUPRC ≈ {y.mean():.4f}", flush=True)

    # per-fold event distribution
    skf = StratifiedKFold(K_FOLDS, shuffle=True, random_state=SEED)
    per_fold = [int(y[te].sum()) for _, te in skf.split(np.zeros(len(y)), y)]
    print(f"[t]   per-fold event counts: {per_fold}", flush=True)
    min_ev = min(per_fold)
    if min_ev < 5:
        print(f"[t]   WARNING: min per-fold event count = {min_ev} — fold-level metrics unstable", flush=True)

    # model variants
    for label, cols, mk in [
        ("rf_full",       FEATS_TAB,    "weighted"),
        ("rf_no_lab",     FEATS_NO_LAB, "weighted"),
        ("rf_oversample", FEATS_TAB,    "oversample"),
    ]:
        print(f"[t]   fitting {label}...", flush=True)
        p, per_fold_ev = _cv_oof(cols, y, mk)
        r = _report(label, y, p)
        r["outcome"] = name
        r["n_events"] = n_events
        r["base_rate"] = float(y.mean())
        r["per_fold_events"] = per_fold_ev
        rows.append(r)
        print(f"[t]     {label:14s} AUROC {r['AUROC']:.3f} ({r['AUROC_lo']:.3f}-{r['AUROC_hi']:.3f})  "
              f"AUPRC {r['AUPRC']:.3f} ({r['AUPRC_lo']:.3f}-{r['AUPRC_hi']:.3f})  Brier {r['Brier']:.4f}",
              flush=True)

    # permutation importance top-5 (only for rf_full, once per outcome)
    if n_events >= 25:
        try:
            top5 = _perm_importance_top5(FEATS_TAB, y, None, "weighted")
            outcome_summary[name] = {
                "n_events": n_events,
                "per_fold_events": per_fold,
                "top5_features": top5.to_dict(),
            }
            print(f"[t]   top-5 permutation-importance features (ΔAUPRC when shuffled):", flush=True)
            for feat, val in top5.items():
                print(f"       {feat:35s} {val:+.4f}", flush=True)
        except Exception as e:
            print(f"[t]   permutation importance failed: {e}", flush=True)
            outcome_summary[name] = {"n_events": n_events, "per_fold_events": per_fold,
                                     "top5_features": None}
    else:
        outcome_summary[name] = {"n_events": n_events, "per_fold_events": per_fold,
                                 "top5_features": "skipped (too few events)"}

# ---------- 5. Save + cross-outcome summary ----------
res = pd.DataFrame(rows)
csv_path = OUT_DIR / "nsqip_full_3outcomes_results.csv"
res.to_csv(csv_path, index=False)
print(f"\n[t] results saved to {csv_path}", flush=True)

print("\n" + "="*60, flush=True)
print("CROSS-OUTCOME SUMMARY", flush=True)
print("="*60, flush=True)
for name, y, no_skill_ref in OUTCOMES:
    sub = res[res["outcome"] == name]
    print(f"\n{name} (n_events={int(y.sum())}, base={y.mean()*100:.2f}%, no-skill AUPRC≈{y.mean():.3f}):", flush=True)
    for _, r in sub.iterrows():
        # flag if AUPRC ~ base rate
        flag_auprc = "  ← AUPRC≈no-skill (no lift)" if r["AUPRC"] < y.mean() * 1.5 else ""
        # flag wide CI
        au_span = r["AUROC_hi"] - r["AUROC_lo"]
        flag_ci = f"  ← wide AUROC CI ({au_span:.2f})" if au_span > 0.20 else ""
        print(f"  {r['model']:14s} AUROC {r['AUROC']:.3f} ({r['AUROC_lo']:.3f}-{r['AUROC_hi']:.3f})  "
              f"AUPRC {r['AUPRC']:.3f} ({r['AUPRC_lo']:.3f}-{r['AUPRC_hi']:.3f})  "
              f"Brier {r['Brier']:.4f}{flag_auprc}{flag_ci}", flush=True)

# Best model per outcome
print("\n[t] Best model per outcome (by AUPRC):", flush=True)
for name, y, no_skill_ref in OUTCOMES:
    sub = res[res["outcome"] == name].sort_values("AUPRC", ascending=False)
    best = sub.iloc[0]
    print(f"  {name:20s} {best['model']:14s} AUROC {best['AUROC']:.3f}  AUPRC {best['AUPRC']:.3f}", flush=True)

with open(OUT_DIR / "nsqip_full_3outcomes_summary.json", "w") as f:
    json.dump(outcome_summary, f, indent=2, default=str)
print(f"[t] summary saved to {OUT_DIR / 'nsqip_full_3outcomes_summary.json'}", flush=True)
