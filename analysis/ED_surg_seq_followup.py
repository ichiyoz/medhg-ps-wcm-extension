"""Follow-up analysis on top of ED_surg_seq.py:
  (a) A3 vs bulk LogID overlap (explains the dropped ED->OR cases)
  (b) ED->OR timing broken down by ED facility (WC vs community)
  (c) 30-day readmission rate: ED->OR cases vs the rest of the cohort
Reads via _read_table(schema=...) only -- no raw-file inspection.
"""
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
a3["Facility"] = _cid.map(_dedup["Facility"])
a3["InTime"]  = pd.to_datetime(a3["InTime"])
a3["OutTime"] = pd.to_datetime(a3["OutTime"])

# ---------------------------------------------------------------------
# (a) LogID overlap between A3 and bulk
# ---------------------------------------------------------------------
a3_ids   = set(a3["LogID"].astype(str))
bulk_ids = set(bulk["LogID"].astype(str))
print("=== (a) A3 vs bulk LogID overlap ===")
print(f"A3 unique LogIDs           : {len(a3_ids):,}")
print(f"bulk unique LogIDs         : {len(bulk_ids):,}")
print(f"in BOTH (inner join)       : {len(a3_ids & bulk_ids):,}")
print(f"A3-only (no features/label): {len(a3_ids - bulk_ids):,}")
print(f"bulk-only (no trajectory)  : {len(bulk_ids - a3_ids):,}")
print()

# ---------------------------------------------------------------------
# rebuild ED->OR pathways (same logic as ED_surg_seq.py)
# ---------------------------------------------------------------------
def ed_to_or_path(grp):
    best = None
    for _, sub in grp.sort_values("SeqInEncounter").groupby("EncounterCSN", sort=False):
        types = sub["UnitType"].tolist()
        if "ED" not in types or "OR" not in types or types.index("ED") >= types.index("OR"):
            continue
        or_pos = types.index("OR")
        ed = sub.iloc[:or_pos]
        ed = ed[ed["UnitType"] == "ED"].iloc[0]
        orr = sub.iloc[or_pos]
        if best is None or ed["InTime"] < best[0]:
            best = (ed["InTime"], ed, orr)
    if best is None:
        return None
    _, ed, orr = best
    return pd.Series({
        "ED_Facility":       ed["Facility"],
        "hrs_arrival_to_or": (orr["InTime"] - ed["InTime"]).total_seconds() / 3600,
    })

paths = a3.groupby("LogID").apply(ed_to_or_path).dropna(how="all")

# ---------------------------------------------------------------------
# (b) ED->OR timing by facility
# ---------------------------------------------------------------------
print("=== (b) ED arrival -> OR start, by ED facility (hours) ===")
g = (paths.groupby("ED_Facility")["hrs_arrival_to_or"]
     .agg(n="size", median="median",
          q25=lambda s: s.quantile(.25), q75=lambda s: s.quantile(.75))
     .sort_values("n", ascending=False))
for fac, r in g.iterrows():
    print(f"  {str(fac):10s} n={int(r['n']):5d}  median {r['median']:5.1f}h  "
          f"IQR {r['q25']:5.1f}-{r['q75']:5.1f}h")
wc_mask = paths["ED_Facility"] == "NYP-WC"
for lab, s in [("WC main", paths.loc[wc_mask, "hrs_arrival_to_or"]),
               ("non-WC", paths.loc[~wc_mask, "hrs_arrival_to_or"])]:
    print(f"  [{lab:8s}] n={len(s):5d}  median {s.median():5.1f}h  "
          f"IQR {s.quantile(.25):5.1f}-{s.quantile(.75):5.1f}h")
print()

# ---------------------------------------------------------------------
# (c) readmission rate: ED->OR vs rest (label lives in bulk)
# ---------------------------------------------------------------------
print("=== (c) 30-day readmission: ED->OR vs rest ===")
ed_or_ids = set(paths.index.astype(str))
lab = bulk[["LogID", "ReadmittedWithin30Days"]].copy()
lab["LogID"] = lab["LogID"].astype(str)
lab["y"] = lab["ReadmittedWithin30Days"].astype(int)
lab["grp"] = np.where(lab["LogID"].isin(ed_or_ids), "ED->OR", "rest")
summary = lab.groupby("grp")["y"].agg(n="size", readmits="sum", rate="mean")
for grp, r in summary.iterrows():
    print(f"  {grp:7s} n={int(r['n']):6,d}  readmits={int(r['readmits']):5,d}  "
          f"rate={r['rate']*100:.2f}%")
overall = lab["y"].mean()
print(f"  overall cohort readmission rate: {overall*100:.2f}%")
print(f"  (ED->OR cases matched to bulk label: "
      f"{lab['LogID'].isin(ed_or_ids).sum():,} of {len(ed_or_ids):,} ED->OR LogIDs)")
