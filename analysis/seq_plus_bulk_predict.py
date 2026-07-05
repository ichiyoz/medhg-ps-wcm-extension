"""Readmission prediction: sequence features vs bulk NSQIP features vs both.

Builds three design matrices on the SAME stratified split so the incremental
lift is directly comparable:
  - seq   : care-unit trajectory structure only (from A3)
  - bulk  : the 40 pre-op NSQIP features (age, ASA, labs, comorbidities, ...)
  - both  : seq + bulk concatenated

Bulk features go through the pipeline's fit_preprocess/apply_preprocess
(impute + one-hot + standardize, fit on TRAIN only -> no leakage). Reads via
_read_table(schema=...) only.
"""
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
from medhg_ps.data import _read_table, fit_preprocess, apply_preprocess

UNITS = ["ED", "Acute", "OR", "Intensive", "Intermediate", "Other"]
MIN_COL_SUPPORT = 50

a3   = _read_table(C.UNIT_EDGES_PARQUET, schema=C.A3_UNIT_EDGES_COLUMNS)
bulk = _read_table(C.ENC_FEATURES_CSV,   schema=C.BULK_FEATURES_COLUMNS)

# --- enterprise unit remap -------------------------------------------
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
a3 = a3.sort_values(["LogID", "InTime"])
seqs = a3.groupby("LogID", sort=False)["UnitType"].apply(list)


def collapse(seq):
    out = []
    for s in seq:
        if not out or out[-1] != s:
            out.append(s)
    return out


# --- sequence features -----------------------------------------------
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
    f["ED_before_OR"]  = int("ED" in c and "OR" in c and c.index("ED") < c.index("OR"))
    f["ICU_after_OR"]  = int("Intensive" in c and "OR" in c
                             and c.index("Intensive") > c.index("OR"))
    f[f"start_{c[0]}"] = 1
    f[f"end_{c[-1]}"]  = 1
    for a, b in zip(c, c[1:]):
        k = f"bg_{a}>{b}"
        f[k] = f.get(k, 0) + 1
    rows.append(f)
F = pd.DataFrame(rows).fillna(0)
seq_always = sorted(c2 for c2 in F.columns
                    if c2.startswith(("cnt_", "n_", "has_", "ED_", "ICU_")))
seq_sparse = [c2 for c2 in F.columns
              if c2 not in set(seq_always) and c2 != "LogID"]

# --- assemble modeling frame (seq + raw bulk features + label) -------
feat_cols_bulk = [c2 for c2 in C.MODEL_FEATURE_COLUMNS if c2 in bulk.columns]
missing = [c2 for c2 in C.MODEL_FEATURE_COLUMNS if c2 not in bulk.columns]
if missing:
    print(f"[warn] bulk features missing from file: {missing}")

bulk2 = bulk[["LogID"] + feat_cols_bulk].copy()
bulk2["LogID"] = bulk2["LogID"].astype(str)
bulk2["y"] = bulk["ReadmittedWithin30Days"].astype(int).values

data = F.merge(bulk2, on="LogID", how="inner").reset_index(drop=True)
y = data["y"].values

idx = np.arange(len(data))
itr, ite = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)
ytr, yte = y[itr], y[ite]

# sparse seq-column support filter computed on TRAIN rows only (no leakage)
seq_keep = [c2 for c2 in seq_sparse
            if (data.iloc[itr][c2] > 0).sum() >= MIN_COL_SUPPORT]
seq_cols = seq_always + seq_keep
print(f"Modeling cohort: {len(data):,} encounters | base rate {y.mean()*100:.2f}%")
print(f"  seq features={len(seq_cols)}  bulk features={len(feat_cols_bulk)}\n")

# bulk block: fit_preprocess on TRAIN rows only, apply to TEST
Xb_tr, state = fit_preprocess(data.loc[itr, feat_cols_bulk], id_cols=[])
Xb_te = apply_preprocess(data.loc[ite, feat_cols_bulk], state)

# seq block: raw numeric arrays
Xs = data[seq_cols].astype(float).values
Xs_tr, Xs_te = Xs[itr], Xs[ite]

SETS = {
    "seq":  (Xs_tr, Xs_te),
    "bulk": (Xb_tr, Xb_te),
    "both": (np.hstack([Xs_tr, Xb_tr]), np.hstack([Xs_te, Xb_te])),
}


def evaluate(Xtr, Xte):
    out = {}
    lr = make_pipeline(StandardScaler(),
                       LogisticRegression(max_iter=3000, class_weight="balanced"))
    lr.fit(Xtr, ytr)
    p = lr.predict_proba(Xte)[:, 1]
    out["LR"] = (roc_auc_score(yte, p), average_precision_score(yte, p))
    hgb = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                         l2_regularization=1.0, random_state=42)
    hgb.fit(Xtr, ytr)
    p = hgb.predict_proba(Xte)[:, 1]
    out["HGB"] = (roc_auc_score(yte, p), average_precision_score(yte, p))
    return out


print(f"=== Held-out AUROC / AUPRC (base rate {yte.mean():.3f}) ===")
print(f"  {'features':6s} {'model':4s}   {'AUROC':>6s}  {'AUPRC':>6s}")
for name, (Xtr, Xte) in SETS.items():
    res = evaluate(Xtr, Xte)
    for model, (auc, ap) in res.items():
        print(f"  {name:6s} {model:4s}   {auc:6.3f}  {ap:6.3f}")
