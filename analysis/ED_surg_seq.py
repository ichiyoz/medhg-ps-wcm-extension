from pathlib import Path

import pandas as pd

import medhg_ps.config as C
from medhg_ps.data import _read_table  # schema-driven read + ID->str coercion

# Header-agnostic reads: column names come from the SQL-derived schema, not
# the file, so these work whether the export carried a header or not.
a3   = _read_table(C.UNIT_EDGES_PARQUET, schema=C.A3_UNIT_EDGES_COLUMNS)
bulk = _read_table(C.ENC_FEATURES_CSV,   schema=C.BULK_FEATURES_COLUMNS)

# The A3 export classifies UnitType via the WC-only #UnitTypeLookup, so every
# regional/community-site department (NYPQ, MIL, BMH, ALN, ...) falls through to
# "Other" -- even ORs and EDs. Remap against the enterprise-wide Unit_Names.xlsx
# (Clarity_ID -> UnitType) to recover them. Granular types collapse to the GNN
# buckets; departments absent from the file keep their original A3 label.
_GNN_BUCKET = {
    "ICU": "Intensive", "PICU": "Intensive", "NICU": "Intensive",
    "Med/Surg": "Acute",
    "Procedural Area": "OR",
    "ED": "ED",
    "Recovery Area": "Intermediate",
}
_units = pd.read_excel(Path(__file__).parent / "Unit_Names.xlsx")
_units["cid"] = _units["Clarity_ID"].astype("Int64").astype(str)
_units["gnn"] = _units["UnitType"].map(_GNN_BUCKET).fillna("Other")
_dedup = (_units.dropna(subset=["Clarity_ID"])
          .drop_duplicates("cid", keep="first")
          .set_index("cid"))
_cid = a3["DepartmentID"].astype(str).str.replace(r"\.0+$", "", regex=True)
a3["UnitType"] = _cid.map(_dedup["gnn"]).fillna(a3["UnitType"])
a3["Facility"] = _cid.map(_dedup["Facility"])  # NYP-WC = Weill Cornell main campus

# Parse stay timestamps for the ED->OR elapsed-time analysis below.
a3["InTime"]  = pd.to_datetime(a3["InTime"])
a3["OutTime"] = pd.to_datetime(a3["OutTime"])

print("A3 UnitType distribution:")
print(a3["UnitType"].value_counts())
print()

# Encounters with any ED visit
has_ed = a3[a3["UnitType"] == "ED"]["LogID"].unique()
total  = a3["LogID"].nunique()
print(f"Encounters with ED in trajectory: {len(has_ed):,} of {total:,} total")
print()

# For each encounter, extract the ED->OR pathway: the facility of the first ED
# that precedes the OR, and the elapsed time from that ED to the OR.
#
# A3 groups ADT events by LogID over a PAT_ID + time window (not by CSN), so a
# single LogID can stitch together separate encounters -- especially via the
# SURGERY_DATE +-30d fallback window when admit/discharge times are NULL. To keep
# only pathways provably within ONE hospitalization, we require the ED and OR to
# share an EncounterCSN: evaluate each CSN separately and take the earliest one
# where an ED precedes an OR.
def ed_to_or_path(grp):
    best = None
    for _, sub in grp.sort_values("SeqInEncounter").groupby("EncounterCSN", sort=False):
        types = sub["UnitType"].tolist()
        if "ED" not in types or "OR" not in types or types.index("ED") >= types.index("OR"):
            continue
        or_pos = types.index("OR")              # first OR within this CSN
        ed = sub.iloc[:or_pos]
        ed = ed[ed["UnitType"] == "ED"].iloc[0]  # first ED before that OR, same CSN
        orr = sub.iloc[or_pos]
        if best is None or ed["InTime"] < best[0]:
            best = (ed["InTime"], ed, orr)
    if best is None:
        return None
    _, ed, orr = best
    return pd.Series({
        "ED_Facility":         ed["Facility"],
        "hrs_arrival_to_or":   (orr["InTime"] - ed["InTime"]).total_seconds() / 3600,
        "hrs_departure_to_or": (orr["InTime"] - ed["OutTime"]).total_seconds() / 3600,
    })

paths    = a3.groupby("LogID").apply(ed_to_or_path).dropna(how="all")
ed_to_or = paths.index
print(f"Encounters where ED precedes OR: {len(ed_to_or):,}")
print()

# WC vs non-WC: facility of the ED that precedes the OR
wc = int((paths["ED_Facility"] == "NYP-WC").sum())
print(f"ED->OR by ED campus: WC (NYP-WC) {wc:,} | non-WC {len(paths) - wc:,}")
print(paths["ED_Facility"].value_counts(dropna=False).to_string())
print()

# Elapsed ED->OR time. Heavy right tail (staged/delayed surgery in long stays),
# so median + IQR are reported alongside the mean.
for col, lab in [("hrs_arrival_to_or",   "ED arrival -> OR start"),
                 ("hrs_departure_to_or", "ED departure -> OR start")]:
    s = paths[col]
    print(f"{lab}: mean {s.mean():.1f} h | median {s.median():.1f} h | "
          f"IQR {s.quantile(.25):.1f}-{s.quantile(.75):.1f} h")
print()

# Cross-reference with bulk features for admission/patient type
# (_read_table already recovered headers and coerced LogID to clean str)
merged = bulk[bulk["LogID"].isin(ed_to_or.astype(str))]

for col in ["PatientType", "HOSP_ADMSN_TYPE_C", "AnesType"]:
    if col in merged.columns:
        print(f"{col} breakdown for ED->OR cases:")
        print(merged[col].value_counts())
        print()