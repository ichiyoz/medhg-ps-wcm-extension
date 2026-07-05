"""Pre-op-only sequence variant + bootstrap CIs on the sequence lift.

Adds a leakage-clean PRE-OP sequence (each trajectory truncated at the first
OR -- only what is known before the knife) and bootstraps the held-out test
set to put 95% CIs on AUROC/AUPRC and on the PAIRED gaps:
    both_full  - bulk   (full trajectory adds ... over tabular)
    both_preop - bulk   (pre-op trajectory alone adds ... over tabular)

Feature sets (all on one stratified split): seq_full, seq_preop, bulk,
both_full, both_preop. Bulk uses fit_preprocess fit on TRAIN only.
Reads via _read_table(schema=...) only.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

import medhg_ps.config as C
from medhg_ps.data import _read_table, fit_preprocess, apply_preprocess

UNITS = ["ED", "Acute", "OR", "Intensive", "Intermediate", "Other"]
MIN_COL_SUPPORT = 50
B = 2000  # bootstrap resamples

a3   = _read_table(C.UNIT_EDGES_PARQUET, schema=C.A3_UNIT_EDGES_COLUMNS)
bulk = _read_table(C.ENC_FEATURES_CSV,   schema=C.BULK_FEATURES_COLUMNS)

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


def featurize(raw_seg):
    """Sequence features from a raw event segment (with repeats)."""
    c = collapse(raw_seg)
    if not c:
        return None
    f = {}
    for u in UNITS:
        f[f"cnt_{u}"] = raw_seg.count(u)
    f["n_events"]      = len(raw_seg)
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
    return f


def build_frame(truncate_at_or):
    rows = []
    for logid, raw in seqs.items():
        seg = raw
        if truncate_at_or and "OR" in raw:
            seg = raw[: raw.index("OR") + 1]   # up to & incl. first OR (pre-op)
        feat = featurize(seg)
        if feat is None:
            continue
        feat["LogID"] = logid
        rows.append(feat)
    # Return ALL candidate columns unpruned; the sparse-support filter is
    # applied later on TRAIN rows only (see select_cols) to avoid leakage.
    return pd.DataFrame(rows).fillna(0)


F_full = build_frame(truncate_at_or=False)
F_pre  = build_frame(truncate_at_or=True).add_prefix("pre_").rename(columns={"pre_LogID": "LogID"})

# --- assemble modeling frame -----------------------------------------
feat_bulk = [c2 for c2 in C.MODEL_FEATURE_COLUMNS if c2 in bulk.columns]
b2 = bulk[["LogID"] + feat_bulk].copy()
b2["LogID"] = b2["LogID"].astype(str)
b2["y"] = bulk["ReadmittedWithin30Days"].astype(int).values

data = (F_full.merge(F_pre, on="LogID", how="inner")
        .merge(b2, on="LogID", how="inner").reset_index(drop=True))
y = data["y"].values
full_cands = [c2 for c2 in F_full.columns if c2 != "LogID"]
pre_cands  = [c2 for c2 in F_pre.columns if c2 != "LogID"]

idx = np.arange(len(data))
itr, ite = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)
ytr, yte = y[itr], y[ite]

# sparse seq-column support filter computed on TRAIN rows only (no leakage)
def select_cols(cands, prefix=""):
    struct = tuple(prefix + s for s in ("cnt_", "n_", "has_", "ED_", "ICU_"))
    always = sorted(c2 for c2 in cands if c2.startswith(struct))
    sparse = [c2 for c2 in cands if c2 not in set(always)]
    keep = [c2 for c2 in sparse
            if (data.iloc[itr][c2] > 0).sum() >= MIN_COL_SUPPORT]
    return always + keep

seq_full_cols = select_cols(full_cands)
seq_pre_cols  = select_cols(pre_cands, prefix="pre_")
print(f"Modeling cohort: {len(data):,} | base {y.mean()*100:.2f}% | "
      f"seq_full={len(seq_full_cols)} seq_preop={len(seq_pre_cols)} bulk={len(feat_bulk)}\n")

Xb_tr, state = fit_preprocess(data.loc[itr, feat_bulk], id_cols=[])
Xb_te = apply_preprocess(data.loc[ite, feat_bulk], state)
Xsf = data[seq_full_cols].astype(float).values
Xsp = data[seq_pre_cols].astype(float).values

SETS = {
    "seq_full":   (Xsf[itr], Xsf[ite]),
    "seq_preop":  (Xsp[itr], Xsp[ite]),
    "bulk":       (Xb_tr, Xb_te),
    "both_full":  (np.hstack([Xsf[itr], Xb_tr]), np.hstack([Xsf[ite], Xb_te])),
    "both_preop": (np.hstack([Xsp[itr], Xb_tr]), np.hstack([Xsp[ite], Xb_te])),
}

# --- fit HGB per set, collect test predictions -----------------------
preds = {}
print(f"=== Point estimates, HGB (base {yte.mean():.3f}) ===")
print(f"  {'features':11s} {'AUROC':>6s} {'AUPRC':>6s}")
for name, (Xtr, Xte) in SETS.items():
    m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                       l2_regularization=1.0, random_state=42)
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    preds[name] = p
    print(f"  {name:11s} {roc_auc_score(yte, p):6.3f} {average_precision_score(yte, p):6.3f}")
print()

# --- bootstrap the test set: CIs + paired diffs ----------------------
rng = np.random.default_rng(0)
n = len(yte)
keys = ["bulk", "both_full", "both_preop"]
au = {k: [] for k in keys}; ap = {k: [] for k in keys}
d_full = {"au": [], "ap": []}; d_pre = {"au": [], "ap": []}
for _ in range(B):
    bi = rng.integers(0, n, n)
    yb = yte[bi]
    if yb.sum() == 0 or yb.sum() == n:
        continue
    cur = {}
    for k in keys:
        a = roc_auc_score(yb, preds[k][bi]); p = average_precision_score(yb, preds[k][bi])
        au[k].append(a); ap[k].append(p); cur[k] = (a, p)
    d_full["au"].append(cur["both_full"][0] - cur["bulk"][0])
    d_full["ap"].append(cur["both_full"][1] - cur["bulk"][1])
    d_pre["au"].append(cur["both_preop"][0] - cur["bulk"][0])
    d_pre["ap"].append(cur["both_preop"][1] - cur["bulk"][1])


def ci(v):
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


print(f"=== Bootstrap 95% CIs ({B} resamples) ===")
for k in keys:
    al, ah = ci(au[k]); pl, ph = ci(ap[k])
    print(f"  {k:11s} AUROC {np.mean(au[k]):.3f} [{al:.3f}, {ah:.3f}] | "
          f"AUPRC {np.mean(ap[k]):.3f} [{pl:.3f}, {ph:.3f}]")
print()
print("=== Paired gaps vs bulk (95% CI; P>0 = fraction of resamples gap>0) ===")
for label, d in [("both_full  - bulk", d_full), ("both_preop - bulk", d_pre)]:
    al, ah = ci(d["au"]); pl, ph = ci(d["ap"])
    pau = np.mean(np.array(d["au"]) > 0); pap = np.mean(np.array(d["ap"]) > 0)
    print(f"  {label}")
    print(f"     dAUROC {np.mean(d['au']):+.4f} [{al:+.4f}, {ah:+.4f}]  P>0={pau:.3f}")
    print(f"     dAUPRC {np.mean(d['ap']):+.4f} [{pl:+.4f}, {ph:+.4f}]  P>0={pap:.3f}")
