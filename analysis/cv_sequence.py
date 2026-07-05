"""CV of the care-unit SEQUENCE model: does the trajectory lift survive?

Single-split work (seq_plus_bulk_predict / seq_preop_bootstrap) found the full
care-unit trajectory adds +0.022 AUPRC over tabular. This re-tests it with
5-fold CV, train-honest feature selection (sparse-support filter on train rows
only), and the pipeline's train-only bulk preprocessing.

Per identical fold, HistGradientBoosting on:
  - bulk : 40 NSQIP tabular features
  - seq  : care-unit sequence features (counts, bigrams, flags, start/end)
  - both : bulk + seq

Reports mean +/- std AUROC/AUPRC and the paired (both - bulk) lift per fold.
Cohort = A3-trajectory INTERSECT bulk (~40.5k). Reads via _read_table only.

    PYTHONPATH=. python analysis/cv_sequence.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import medhg_ps.config as C
from medhg_ps.data import _read_table, fit_preprocess, apply_preprocess

K, SEED, MIN_COL_SUPPORT = 5, 42, 50
UNITS = ["ED", "Acute", "OR", "Intensive", "Intermediate", "Other"]

a3   = _read_table(C.UNIT_EDGES_PARQUET, schema=C.A3_UNIT_EDGES_COLUMNS)
bulk = _read_table(C.ENC_FEATURES_CSV,   schema=C.BULK_FEATURES_COLUMNS)

# --- enterprise unit remap (same as the seq_* scripts) ---------------
_GNN_BUCKET = {"ICU": "Intensive", "PICU": "Intensive", "NICU": "Intensive",
               "Med/Surg": "Acute", "Procedural Area": "OR", "ED": "ED",
               "Recovery Area": "Intermediate"}
_u = pd.read_excel(Path(__file__).parent / "Unit_Names.xlsx")
_u["cid"] = _u["Clarity_ID"].astype("Int64").astype(str)
_u["gnn"] = _u["UnitType"].map(_GNN_BUCKET).fillna("Other")
_d = _u.dropna(subset=["Clarity_ID"]).drop_duplicates("cid", keep="first").set_index("cid")
_cid = a3["DepartmentID"].astype(str).str.replace(r"\.0+$", "", regex=True)
a3["UnitType"] = _cid.map(_d["gnn"]).fillna(a3["UnitType"])
a3["InTime"] = pd.to_datetime(a3["InTime"])
a3["LogID"] = a3["LogID"].astype(str)
a3 = a3.sort_values(["LogID", "InTime"])
seqs = a3.groupby("LogID", sort=False)["UnitType"].apply(list)


def collapse(seq):
    out = []
    for s in seq:
        if not out or out[-1] != s:
            out.append(s)
    return out


rows = []
for logid, raw in seqs.items():
    c = collapse(raw)
    f = {"LogID": logid}
    for u in UNITS:
        f[f"cnt_{u}"] = raw.count(u)
    f["n_events"] = len(raw); f["n_stops"] = len(c)
    f["n_transitions"] = max(len(c) - 1, 0); f["n_or_visits"] = c.count("OR")
    f["has_ED"] = int("ED" in c); f["has_ICU"] = int("Intensive" in c)
    f["ED_before_OR"] = int("ED" in c and "OR" in c and c.index("ED") < c.index("OR"))
    f["ICU_after_OR"] = int("Intensive" in c and "OR" in c and c.index("Intensive") > c.index("OR"))
    f[f"start_{c[0]}"] = 1; f[f"end_{c[-1]}"] = 1
    for a, b in zip(c, c[1:]):
        k = f"bg_{a}>{b}"; f[k] = f.get(k, 0) + 1
    rows.append(f)
F = pd.DataFrame(rows).fillna(0)
seq_always = sorted(c for c in F.columns if c.startswith(("cnt_", "n_", "has_", "ED_", "ICU_")))
seq_sparse = [c for c in F.columns if c not in set(seq_always) and c != "LogID"]

feat_bulk = [c for c in C.MODEL_FEATURE_COLUMNS if c in bulk.columns]
b2 = bulk[["LogID"] + feat_bulk].copy()
b2["LogID"] = b2["LogID"].astype(str)
b2["y"] = bulk["ReadmittedWithin30Days"].astype(int).values
data = F.merge(b2, on="LogID", how="inner").reset_index(drop=True)
y = data["y"].values
N = len(data)
print(f"[cv] cohort (A3 ∩ bulk) = {N:,}  base = {y.mean()*100:.2f}%  "
      f"bulk feats={len(feat_bulk)}  seq candidates={len(seq_sparse)+len(seq_always)}", flush=True)


def _hgb():
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                          l2_regularization=1.0, random_state=42)


def metrics(p, te):
    return roc_auc_score(y[te], p), average_precision_score(y[te], p)


skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
MODELS = ["bulk", "seq", "both"]
res = {m: {"auroc": [], "auprc": []} for m in MODELS}

for fold, (tr, te) in enumerate(skf.split(np.zeros(N), y)):
    # train-only sparse-support filter
    keep = [c for c in seq_sparse if (data.iloc[tr][c] > 0).sum() >= MIN_COL_SUPPORT]
    seq_cols = seq_always + keep
    Xs = data[seq_cols].astype(float).values

    # train-only bulk preprocessing
    Xb_tr, st = fit_preprocess(data.loc[tr, feat_bulk], id_cols=[])
    Xb_te = apply_preprocess(data.loc[te, feat_bulk], st)

    SETS = {
        "bulk": (Xb_tr, Xb_te),
        "seq":  (Xs[tr], Xs[te]),
        "both": (np.hstack([Xs[tr], Xb_tr]), np.hstack([Xs[te], Xb_te])),
    }
    line = {}
    for m, (Xtr, Xte) in SETS.items():
        clf = _hgb(); clf.fit(Xtr, y[tr])
        a, p = metrics(clf.predict_proba(Xte)[:, 1], te)
        res[m]["auroc"].append(a); res[m]["auprc"].append(p); line[m] = (a, p)
    print(f"[cv] fold {fold+1}/{K}  bulk AUPRC={line['bulk'][1]:.3f}  "
          f"both AUPRC={line['both'][1]:.3f}  "
          f"(both-bulk AUROC {line['both'][0]-line['bulk'][0]:+.3f} / "
          f"AUPRC {line['both'][1]-line['bulk'][1]:+.3f})", flush=True)

print(f"\n=== {K}-fold CV (mean +/- std) | base {y.mean()*100:.2f}% ===")
print(f"  {'model':6s} {'AUROC':>16s} {'AUPRC':>16s}")
for m in MODELS:
    au, ap = np.array(res[m]["auroc"]), np.array(res[m]["auprc"])
    print(f"  {m:6s} {au.mean():.3f} +/- {au.std():.3f}    {ap.mean():.3f} +/- {ap.std():.3f}")

da = np.array(res["both"]["auroc"]) - np.array(res["bulk"]["auroc"])
dp = np.array(res["both"]["auprc"]) - np.array(res["bulk"]["auprc"])
print(f"\n=== Paired lift  both - bulk ===")
print(f"  dAUROC mean {da.mean():+.4f} +/- {da.std():.4f}  (folds>0 {int((da>0).sum())}/{K})  {np.round(da,3).tolist()}")
print(f"  dAUPRC mean {dp.mean():+.4f} +/- {dp.std():.4f}  (folds>0 {int((dp>0).sum())}/{K})  {np.round(dp,3).tolist()}")
