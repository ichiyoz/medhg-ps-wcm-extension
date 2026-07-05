"""Process mining of the ED->OR group + inpatient-stratified readmission test.

Part 1 (process mining): treat each ED->OR hospitalization (the CSN where an
ED precedes the OR) as a process case, UnitType as the activity, ordered by
SeqInEncounter. Reports:
  - trace-variant frequencies (the actual care pathways) + per-variant
    30-day readmission rate
  - the directly-follows graph (unit-to-unit transition counts)
  - start/end activities and trace-length distribution

Part 2: confirm the ED->OR readmission effect holds WITHIN inpatients only
(PatientType == 'I'), with a two-proportion z-test and risk ratio.

Dependency-free (pandas + stdlib). Reads via _read_table(schema=...) only.
"""
from collections import Counter
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

import medhg_ps.config as C
from medhg_ps.data import _read_table

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

# readmission label lookup (LogID -> 0/1), only for cases present in bulk
lab = bulk[["LogID", "PatientType", "ReadmittedWithin30Days"]].copy()
lab["LogID"] = lab["LogID"].astype(str)
lab["y"] = lab["ReadmittedWithin30Days"].astype(int)
y_by_log    = dict(zip(lab["LogID"], lab["y"]))
ptype_by_log = dict(zip(lab["LogID"], lab["PatientType"]))


def collapse(seq):
    """Drop consecutive duplicate activities (remove self-loops)."""
    out = []
    for s in seq:
        if not out or out[-1] != s:
            out.append(s)
    return out


# ---------------------------------------------------------------------
# Extract one process trace per ED->OR LogID: the CSN where ED first
# precedes OR, as the full ordered (collapsed) UnitType sequence.
# ---------------------------------------------------------------------
traces = []   # list of (LogID, trace_tuple)
for logid, grp in a3.sort_values("SeqInEncounter").groupby("LogID", sort=False):
    best = None  # (ed_intime, trace)
    for _, sub in grp.groupby("EncounterCSN", sort=False):
        types = sub["UnitType"].tolist()
        if "ED" not in types or "OR" not in types:
            continue
        if types.index("ED") >= types.index("OR"):
            continue
        ed_time = sub.loc[sub["UnitType"] == "ED", "InTime"].iloc[0]
        if best is None or ed_time < best[0]:
            best = (ed_time, collapse(types))
    if best is not None:
        traces.append((str(logid), tuple(best[1])))

n_cases = len(traces)
print(f"=== ED->OR process mining: {n_cases:,} hospitalizations (cases) ===\n")

# ---- trace-length distribution ----
lengths = pd.Series([len(t) for _, t in traces])
print("Trace length (distinct units after collapsing self-loops):")
print(f"  mean {lengths.mean():.2f} | median {int(lengths.median())} | "
      f"min {lengths.min()} | max {lengths.max()}\n")

# ---- variant frequencies + per-variant readmission ----
var_cases = Counter(t for _, t in traces)
# readmission per variant (only over LogIDs that have a bulk label)
var_y_sum, var_y_n = Counter(), Counter()
for logid, t in traces:
    if logid in y_by_log:
        var_y_sum[t] += y_by_log[logid]
        var_y_n[t]   += 1

print("Top 15 care pathways (variants):")
print(f"  {'#':>5} {'%':>5} {'readmit%':>9}  pathway")
for t, cnt in var_cases.most_common(15):
    n_lab = var_y_n[t]
    rr = (var_y_sum[t] / n_lab * 100) if n_lab else float("nan")
    print(f"  {cnt:5d} {cnt/n_cases*100:4.1f}% {rr:8.1f}%  "
          + " -> ".join(t))
print(f"\n  ({len(var_cases):,} distinct variants total)\n")

# ---- directly-follows graph (transition counts) ----
dfg = Counter()
for _, t in traces:
    for a, b in zip(t, t[1:]):
        dfg[(a, b)] += 1
print("Directly-follows graph (top 15 transitions):")
for (a, b), cnt in dfg.most_common(15):
    print(f"  {a:12s} -> {b:12s} {cnt:6d}")
print()

# ---- start / end activities ----
starts = Counter(t[0]  for _, t in traces)
ends   = Counter(t[-1] for _, t in traces)
print("Start activities:", dict(starts.most_common()))
print("End activities:  ", dict(ends.most_common()))
print()

# ---------------------------------------------------------------------
# Part 2: ED->OR readmission effect WITHIN inpatients only
# ---------------------------------------------------------------------
def two_prop_z(x1, n1, x2, n2):
    p1, p2 = x1 / n1, x2 / n2
    p = (x1 + x2) / (n1 + n2)
    se = sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    pval = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return z, pval


ed_or_ids = {logid for logid, _ in traces}
lab["is_ed_or"] = lab["LogID"].isin(ed_or_ids)

print("=== Part 2: readmission by ED->OR, stratified by PatientType ===")
for ptype, name in [("I", "INPATIENTS only"), ("O", "OUTPATIENTS only")]:
    sub = lab[lab["PatientType"] == ptype]
    g = sub.groupby("is_ed_or")["y"].agg(n="size", x="sum")
    if True not in g.index or False not in g.index:
        print(f"\n[{name}] insufficient groups"); continue
    x1, n1 = int(g.loc[True, "x"]),  int(g.loc[True, "n"])   # ED->OR
    x0, n0 = int(g.loc[False, "x"]), int(g.loc[False, "n"])  # non-ED->OR
    r1, r0 = x1 / n1, x0 / n0
    z, pval = two_prop_z(x1, n1, x0, n0)
    print(f"\n[{name}]")
    print(f"  ED->OR     : {x1:5d}/{n1:6d} = {r1*100:5.2f}%")
    print(f"  non-ED->OR : {x0:5d}/{n0:6d} = {r0*100:5.2f}%")
    print(f"  risk ratio : {r1/r0:.2f}x   (abs diff {(r1-r0)*100:+.2f} pp)")
    print(f"  two-prop z = {z:.2f}, p = {pval:.2e}")
