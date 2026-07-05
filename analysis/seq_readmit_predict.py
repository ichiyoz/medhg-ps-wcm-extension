"""Predict 30-day readmission from the care-unit SEQUENCE alone (A3 trajectory).

Each encounter's ordered UnitType trajectory is turned into structural
sequence features -- no labs, no comorbidities -- to see how much signal the
pathway shape alone carries:
  - per-unit visit counts (ED, Acute, OR, Intensive, ...)
  - directly-follows bigrams (e.g. OR>Intensive = post-op ICU)
  - structural flags (has_ED, ED_before_OR, ICU_after_OR, #OR visits, length)
  - start / end activity

Trains LogisticRegression (interpretable) + HistGradientBoosting on a
stratified 80/20 split; reports AUROC / AUPRC vs the base rate and the top
sequence features. Reads via _read_table(schema=...) only.

Scoring-time note: the prediction point is post-surgery / pre-discharge, so
the in-hospital trajectory (incl. post-op units) is known. No future info.
"""
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import medhg_ps.config as C
from medhg_ps.data import _read_table

UNITS = ["ED", "Acute", "OR", "Intensive", "Intermediate", "Other"]
MIN_COL_SUPPORT = 50      # drop bigram/start/end features rarer than this

a3   = _read_table(C.UNIT_EDGES_PARQUET, schema=C.A3_UNIT_EDGES_COLUMNS)
bulk = _read_table(C.ENC_FEATURES_CSV,   schema=C.BULK_FEATURES_COLUMNS)

# --- enterprise unit remap (same as ED_surg_seq.py) -------------------
_GNN_BUCKET = {
    "ICU": "Intensive", "PICU": "Intensive", "NICU": "Intensive",
    "Med/Surg": "Acute", "Procedural Area": "OR", "ED": "ED",
    "Recovery Area": "Intermediate",
}
_units = pd.read_excel(Path(__file__).parent / "Unit_Names.xlsx")
_units["cid"] = _units["Clarity_ID"].astype("Int64").astype(str)
_units["gnn"] = _units["UnitType"].map(_GNN_BUCKET).fillna("Other")
_dedup = (_units.dropna(subset=["Clarity_ID"])
          .drop_duplicates("cid", keep="first").set_index("cid"))
_cid = a3["DepartmentID"].astype(str).str.replace(r"\.0+$", "", regex=True)
a3["UnitType"] = _cid.map(_dedup["gnn"]).fillna(a3["UnitType"])
a3["InTime"]   = pd.to_datetime(a3["InTime"])
a3["LogID"]    = a3["LogID"].astype(str)

# order each encounter's events in time, build the unit sequence
a3 = a3.sort_values(["LogID", "InTime"])
seqs = a3.groupby("LogID", sort=False)["UnitType"].apply(list)


def collapse(seq):
    out = []
    for s in seq:
        if not out or out[-1] != s:
            out.append(s)
    return out


# --- featurize each sequence -----------------------------------------
rows = []
for logid, raw in seqs.items():
    c = collapse(raw)
    f = {"LogID": logid}
    for u in UNITS:
        f[f"cnt_{u}"] = raw.count(u)
    f["n_events"]      = len(raw)
    f["n_stops"]       = len(c)
    f["n_transitions"] = max(len(c) - 1, 0)
    f["n_or_visits"]   = c.count("OR")
    f["has_ED"]        = int("ED" in c)
    f["has_ICU"]       = int("Intensive" in c)
    f["ED_before_OR"]  = int("ED" in c and "OR" in c
                             and c.index("ED") < c.index("OR"))
    f["ICU_after_OR"]  = int("Intensive" in c and "OR" in c
                             and c.index("Intensive") > c.index("OR"))
    f[f"start_{c[0]}"]  = 1
    f[f"end_{c[-1]}"]   = 1
    for a, b in zip(c, c[1:]):
        k = f"bg_{a}>{b}"
        f[k] = f.get(k, 0) + 1
    rows.append(f)

F = pd.DataFrame(rows).fillna(0)

# --- attach label, restrict to encounters present in bulk ------------
lab = bulk[["LogID"]].copy()
lab["LogID"] = lab["LogID"].astype(str)
lab["y"] = bulk["ReadmittedWithin30Days"].astype(int).values
data = F.merge(lab, on="LogID", how="inner").reset_index(drop=True)
y = data["y"].values

# Split FIRST, then select features: the sparse-column support filter is
# computed on TRAIN rows only so no test-set information leaks into which
# columns are kept. `always` is a sorted list (not a set) for deterministic
# feature order / reproducible coefficient reporting.
idx = np.arange(len(data))
itr, ite = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)

always = sorted(c2 for c2 in F.columns
                if c2.startswith(("cnt_", "n_", "has_", "ED_", "ICU_")))
sparse = [c2 for c2 in F.columns if c2 not in set(always) and c2 != "LogID"]
keep = [c2 for c2 in sparse
        if (data.iloc[itr][c2] > 0).sum() >= MIN_COL_SUPPORT]
feat_cols = always + keep

X = data[feat_cols].astype(float).values
print(f"Modeling cohort (trajectory + label): {len(data):,} encounters, "
      f"{len(feat_cols)} sequence features")
print(f"Base readmission rate: {y.mean()*100:.2f}%\n")

Xtr, Xte, ytr, yte = X[itr], X[ite], y[itr], y[ite]

# --- models ----------------------------------------------------------
lr = make_pipeline(StandardScaler(),
                   LogisticRegression(max_iter=2000, class_weight="balanced"))
lr.fit(Xtr, ytr)
p_lr = lr.predict_proba(Xte)[:, 1]

hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                     l2_regularization=1.0, random_state=42)
hgb.fit(Xtr, ytr)
p_hgb = hgb.predict_proba(Xte)[:, 1]

print("=== Held-out test performance (sequence features only) ===")
for name, p in [("LogisticRegression", p_lr),
                ("HistGradientBoosting", p_hgb)]:
    print(f"  {name:22s} AUROC {roc_auc_score(yte, p):.3f} | "
          f"AUPRC {average_precision_score(yte, p):.3f} "
          f"(base {yte.mean():.3f})")
print()

# --- most predictive sequence features (standardized LR coefs) -------
coefs = lr.named_steps["logisticregression"].coef_[0]
order = np.argsort(coefs)
print("Top sequence features RAISING readmission risk:")
for i in order[::-1][:12]:
    print(f"  {coefs[i]:+.3f}  {feat_cols[i]}")
print("\nTop sequence features LOWERING readmission risk:")
for i in order[:8]:
    print(f"  {coefs[i]:+.3f}  {feat_cols[i]}")
