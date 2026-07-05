"""CV of the pre-op vs post-op sequence lift -- the mechanistic test.

Confirms (on 5-fold CV, fair footing) that the care-unit trajectory's
readmission signal is POST-operative: the full trajectory adds AUPRC over
tabular, but truncating it at the first OR (pre-op only) removes the lift.

Per identical fold, HistGradientBoosting on:
  - bulk        : tabular features
  - both_full   : tabular + full-trajectory sequence features
  - both_preop  : tabular + sequence truncated at first OR (pre-op only)

Train-honest sparse-support filter (train rows only) + train-only bulk
preprocessing. Cohort = A3 INTERSECT bulk. Reads via _read_table only.

    PYTHONPATH=. python analysis/cv_sequence_preop.py
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


def featurize(raw_seg):
    c = collapse(raw_seg)
    if not c:
        return None
    f = {}
    for u in UNITS:
        f[f"cnt_{u}"] = raw_seg.count(u)
    f["n_events"] = len(raw_seg); f["n_stops"] = len(c)
    f["n_transitions"] = max(len(c) - 1, 0); f["n_or_visits"] = c.count("OR")
    f["has_ED"] = int("ED" in c); f["has_ICU"] = int("Intensive" in c)
    f["ED_before_OR"] = int("ED" in c and "OR" in c and c.index("ED") < c.index("OR"))
    f["ICU_after_OR"] = int("Intensive" in c and "OR" in c and c.index("Intensive") > c.index("OR"))
    f[f"start_{c[0]}"] = 1; f[f"end_{c[-1]}"] = 1
    for a, b in zip(c, c[1:]):
        k = f"bg_{a}>{b}"; f[k] = f.get(k, 0) + 1
    return f


def build_frame(truncate_at_or):
    rows = []
    for logid, raw in seqs.items():
        seg = raw[: raw.index("OR") + 1] if (truncate_at_or and "OR" in raw) else raw
        feat = featurize(seg)
        if feat is None:
            continue
        feat["LogID"] = logid
        rows.append(feat)
    return pd.DataFrame(rows).fillna(0)


F_full = build_frame(False)
F_pre  = build_frame(True).add_prefix("pre_").rename(columns={"pre_LogID": "LogID"})

feat_bulk = [c for c in C.MODEL_FEATURE_COLUMNS if c in bulk.columns]
b2 = bulk[["LogID"] + feat_bulk].copy()
b2["LogID"] = b2["LogID"].astype(str)
b2["y"] = bulk["ReadmittedWithin30Days"].astype(int).values
data = (F_full.merge(F_pre, on="LogID", how="inner")
        .merge(b2, on="LogID", how="inner").reset_index(drop=True))
y = data["y"].values
N = len(data)
full_cands = [c for c in F_full.columns if c != "LogID"]
pre_cands  = [c for c in F_pre.columns if c != "LogID"]
print(f"[cv] cohort (A3 ∩ bulk) = {N:,}  base = {y.mean()*100:.2f}%  bulk feats={len(feat_bulk)}", flush=True)


def select_cols(cands, tr, prefix=""):
    struct = tuple(prefix + s for s in ("cnt_", "n_", "has_", "ED_", "ICU_"))
    always = sorted(c for c in cands if c.startswith(struct))
    sparse = [c for c in cands if c not in set(always)]
    keep = [c for c in sparse if (data.iloc[tr][c] > 0).sum() >= MIN_COL_SUPPORT]
    return always + keep


def _hgb():
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                          l2_regularization=1.0, random_state=42)


skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
MODELS = ["bulk", "both_full", "both_preop"]
res = {m: {"auroc": [], "auprc": []} for m in MODELS}

for fold, (tr, te) in enumerate(skf.split(np.zeros(N), y)):
    Xb_tr, st = fit_preprocess(data.loc[tr, feat_bulk], id_cols=[])
    Xb_te = apply_preprocess(data.loc[te, feat_bulk], st)
    full_cols = select_cols(full_cands, tr)
    pre_cols  = select_cols(pre_cands, tr, prefix="pre_")
    Xf = data[full_cols].astype(float).values
    Xp = data[pre_cols].astype(float).values
    SETS = {
        "bulk":       (Xb_tr, Xb_te),
        "both_full":  (np.hstack([Xf[tr], Xb_tr]), np.hstack([Xf[te], Xb_te])),
        "both_preop": (np.hstack([Xp[tr], Xb_tr]), np.hstack([Xp[te], Xb_te])),
    }
    ln = {}
    for m, (Xtr, Xte) in SETS.items():
        clf = _hgb(); clf.fit(Xtr, y[tr])
        p = clf.predict_proba(Xte)[:, 1]
        a, pr = roc_auc_score(y[te], p), average_precision_score(y[te], p)
        res[m]["auroc"].append(a); res[m]["auprc"].append(pr); ln[m] = (a, pr)
    print(f"[cv] fold {fold+1}/{K}  "
          f"full-bulk AUPRC {ln['both_full'][1]-ln['bulk'][1]:+.3f}  "
          f"preop-bulk AUPRC {ln['both_preop'][1]-ln['bulk'][1]:+.3f}", flush=True)

print(f"\n=== {K}-fold CV (mean +/- std) | base {y.mean()*100:.2f}% ===")
print(f"  {'model':11s} {'AUROC':>16s} {'AUPRC':>16s}")
for m in MODELS:
    au, ap = np.array(res[m]["auroc"]), np.array(res[m]["auprc"])
    print(f"  {m:11s} {au.mean():.3f} +/- {au.std():.3f}    {ap.mean():.3f} +/- {ap.std():.3f}")

for m in ["both_full", "both_preop"]:
    dp = np.array(res[m]["auprc"]) - np.array(res["bulk"]["auprc"])
    da = np.array(res[m]["auroc"]) - np.array(res["bulk"]["auroc"])
    print(f"\n  {m} - bulk:  dAUPRC {dp.mean():+.4f} +/- {dp.std():.4f} "
          f"(folds>0 {int((dp>0).sum())}/{K})  |  dAUROC {da.mean():+.4f} "
          f"(folds>0 {int((da>0).sum())}/{K})")
