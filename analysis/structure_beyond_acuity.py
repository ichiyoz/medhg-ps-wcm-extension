"""Rigorous test: does RAW care-structure carry readmission signal BEYOND
acuity/case-mix? Nested 5-fold CV (BASE vs BASE+STRUCTURE) for HGB and LR, plus
an in-sample MLE-logistic nested likelihood-ratio test (chi-square, df=#structure).

NOT using DICE cluster membership (that is outcome-fit by construction -> circular).
Care-structure = raw unit composition (A3 14-bucket InstitutionType), care-path
trajectory (preop LOS / #units / %ICU / %ED / OR-only), and provider-team
composition (A2+A4 ProvType counts, surgeon CaseVolume2yr).

    PYTHONPATH=. python analysis/structure_beyond_acuity.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy.stats import chi2
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

import medhg_ps.config as C
from medhg_ps.deploy import assemble_training_frame
from medhg_ps.data import load_raw

merged, feat_cols, cpt_arr, Fseq, seq_all, y = assemble_training_frame()
raw = load_raw()
y = np.asarray(y).astype(int); N = len(merged)
lids = merged["LogID"].astype(str)
INST = ["Procedural Area", "Med/Surg", "ICU", "ED", "PICU",
        "Labor and Delivery", "Postpartum", "Pediatrics", "Antepartum", "Nursery"]
PROV = ["Attending", "Resident", "Fellow", "Anesthesiologist", "Nurse Anesthetist"]
COMORB = ["Diabetes Mellitus", "Current Smoker within 1 year", "Ventilator Dependent",
          "History of Severe COPD", "Ascites", "Heart Failure", "Hypertension requiring medication",
          "Preop Acute Kidney Injury", "Preop Dialysis", "Disseminated Cancer",
          "Immunosuppressive Therapy", "Bleeding Disorder", "Preop RBC Transfusions (72h)"]
EVENTS = ["# of Cardiac Arrest Requiring CPR", "# of Stroke/Cerebral Vascular Acccident (CVA)",
          "# of Postop Unplanned Intubation"]
YES = {"yes", "1", "1.0", "true", "insulin", "non-insulin"}


def num(c): return pd.to_numeric(merged[c], errors="coerce")


# ---- BASE: acuity / case-mix ----
B = pd.DataFrame(index=range(N))
B["ASA"] = num("ASAClass"); B["inpatient"] = (merged["PatientType"].astype(str) == "I").astype(float)
B["age"] = num("AgeYears"); B["optime"] = num("CutToClose")
B["n_other"] = num("# of Other Procedures"); B["n_conc"] = num("# of Concurrent Procedures")
for c in COMORB:
    B["cm_" + c[:14]] = merged[c].astype(str).str.strip().str.lower().isin(YES).astype(float)
for c in EVENTS:
    B["ev_" + c[8:20]] = num(c).fillna(0).clip(0, 1)
cpt = merged["PrimaryCPT"].astype(str)
for cc in cpt.value_counts().head(40).index:          # top-40 CPT dummies (in BOTH models)
    B[f"cpt_{cc}"] = (cpt == cc).astype(float)

# ---- STRUCTURE: units + trajectory + team ----
a3 = raw.enc_unit_edges.copy(); a3["LogID"] = a3["LogID"].astype(str)
a3["InTime"] = pd.to_datetime(a3["InTime"], errors="coerce")
a3 = a3.sort_values(["LogID", "SeqInEncounter", "InTime"])
seqs = a3.groupby("LogID", sort=False)["UnitType"].apply(list)
inst = a3.groupby("LogID")["InstitutionType"].apply(lambda s: set(s.dropna()))
S = pd.DataFrame(index=range(N))
for cat in INST:
    S["u_" + cat[:10]] = lids.map(inst.apply(lambda st, c=cat: int(c in st))).fillna(0).values
def collapse(s):
    o = []
    for x in s:
        if not o or o[-1] != x: o.append(x)
    return o
S["has_ICU"] = lids.map(seqs.apply(lambda s: int("Intensive" in s))).fillna(0).values
S["has_ED"] = lids.map(seqs.apply(lambda s: int("ED" in s))).fillna(0).values
S["OR_only"] = lids.map(seqs.apply(lambda s: int(set(collapse(s)) == {"OR"}))).fillna(0).values
S["n_units"] = lids.map(seqs.apply(lambda s: len(set(s)))).fillna(0).values
for c in C.TRAJECTORY_FEATURE_COLUMNS:                 # preop_los_*, preop_transfer_count, preop_n_units
    S[c] = num(c).fillna(0).values if c in merged else 0.0
pa = raw.enc_prov_edges[["LogID", "ProvID"]].assign(LogID=lambda d: d["LogID"].astype(str))
pa["ptype"] = pa["ProvID"].map(raw.prov_attrs.set_index("ProvID")["ProvType"])
comp = pa.pivot_table(index="LogID", columns="ptype", values="ProvID", aggfunc="count", fill_value=0)
for role in PROV:
    S["t_" + role.split()[0][:6]] = lids.map(comp[role]).fillna(0).values if role in comp else 0.0
S["n_prov"] = lids.map(pa.groupby("LogID")["ProvID"].nunique()).fillna(0).values
vol = pd.to_numeric(raw.prov_attrs.set_index("ProvID")["CaseVolume2yr"], errors="coerce")
S["surg_vol"] = lids.map(raw.encounters.assign(LogID=lambda d: d["LogID"].astype(str))
                         .set_index("LogID")["PrimarySurgID"]).map(vol).fillna(vol.median()).values

B = B.fillna(B.median()); S = S.fillna(S.median())
print(f"cohort {N:,}  base {y.mean()*100:.2f}%  |  BASE feats={B.shape[1]}  STRUCTURE feats={S.shape[1]}", flush=True)
Xb, Xf = B.values, np.hstack([B.values, S.values])


# ---- nested 5-fold CV ----
def cv(X):
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    hg = {"au": [], "ap": []}; lr = {"au": [], "ap": []}
    for tr, te in skf.split(np.zeros(N), y):
        m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                            l2_regularization=1.0, random_state=42).fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        hg["au"].append(roc_auc_score(y[te], p)); hg["ap"].append(average_precision_score(y[te], p))
        sc = StandardScaler().fit(X[tr])
        ml = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
        q = ml.predict_proba(sc.transform(X[te]))[:, 1]
        lr["au"].append(roc_auc_score(y[te], q)); lr["ap"].append(average_precision_score(y[te], q))
    return hg, lr

hb, lb = cv(Xb); hf, lf = cv(Xf)
print("\n=== nested 5-fold CV: BASE (acuity/case-mix) vs BASE+STRUCTURE ===")
for name, base, full in [("HGB", hb, hf), ("LR", lb, lf)]:
    dau = np.array(full["au"]) - np.array(base["au"]); dap = np.array(full["ap"]) - np.array(base["ap"])
    print(f"  {name}:  BASE AUROC {np.mean(base['au']):.3f} AUPRC {np.mean(base['ap']):.3f}  ->  "
          f"+STRUCT AUROC {np.mean(full['au']):.3f} AUPRC {np.mean(full['ap']):.3f}")
    print(f"        paired dAUROC {dau.mean():+.4f} (folds>0 {int((dau>0).sum())}/5)   "
          f"dAUPRC {dap.mean():+.4f} (folds>0 {int((dap>0).sum())}/5)")

# ---- in-sample MLE-logistic nested likelihood-ratio test ----
def ll(X):
    sc = StandardScaler().fit(X)
    m = LogisticRegression(penalty=None, max_iter=5000).fit(sc.transform(X), y)
    p = np.clip(m.predict_proba(sc.transform(X))[:, 1], 1e-9, 1 - 1e-9)
    return float((y * np.log(p) + (1 - y) * np.log(1 - p)).sum())
llb, llf = ll(Xb), ll(Xf)
G = 2 * (llf - llb); dfree = S.shape[1]; p = chi2.sf(G, dfree)
print(f"\n=== nested likelihood-ratio test (in-sample MLE logistic) ===")
print(f"  LL(BASE)={llb:.1f}  LL(BASE+STRUCT)={llf:.1f}  G=2ΔLL={G:.1f}  df={dfree}  p={p:.2e}")
print(f"  -> care STRUCTURE {'DOES' if p < 0.05 else 'does NOT'} add readmission signal beyond acuity/case-mix "
      f"(chi-square, df={dfree})")
