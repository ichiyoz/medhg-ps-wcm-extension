"""Classic LACE clinical-score baseline for readmission (WCM SQL-fixed cohort).

van Walraven CMAJ 2010:
    L: length of stay        0-7 pts (bins: 0/1/2/3/4-6/7-13/>=14 -> 0/1/2/3/4/5/7)
    A: emergent admission    0 or 3 pts
    C: Charlson comorbidity  0-5 pts (weighted sum from ICD-10, capped at 5)
    E: ED visits in past 6mo 0-4 pts (min(count, 4))
Total 0-19; >=10 typically flagged high-risk.

WCM adaptation. Two components (L, A) are derivable from the current
bulk file + A3 unit trajectory. Two (C, E) require a Clarity SQL pull:

    L: derived from A3 (max OutTime - min InTime).
    A: TRUE emergent flag via A3 first-unit == ED. Optional fallback to
       PatientType (Inpatient=3, Outpatient=0) — historically produces a
       slightly HIGHER discrimination on this MIS cohort because it
       catches urgent-not-emergent inpatients too. Both parameterizations
       are supported via the emergent_source= argument.
    C: full Charlson requires sql/LACE_Components.sql (index-encounter
       ICD-10 -> weighted Charlson categories). If lace_components.csv
       is not present, falls back to a partial Charlson computed from
       the WCM comorbidity flags in bulk features.
    E: full 180-day prior-ED-visit count from sql/LACE_Components.sql.
       Falls back to 0 for every patient if the file is not present.

Runs are labeled with the components that were computed from real data
vs. approximated / dropped, so results are unambiguous in the manuscript.

Usage:
    python -m analysis.lace_baseline                    # falls back to approximations
    # After running sql/LACE_Components.sql and saving output to
    #   ~/Downloads/medhg_ps_data/lace_components.csv
    python -m analysis.lace_baseline
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import medhg_ps.config as C
import medhg_ps.data as d
from medhg_ps.deploy import assemble_training_frame
from medhg_ps.evaluate import _bootstrap_ci

# Charlson weights (Deyo/Sundararajan)
CHARLSON_WEIGHTS = dict(
    mi=1, chf=1, pvd=1, cvd=1, dementia=1, copd=1, rheum=1, pud=1,
    liver_mild=1, dm_uncomp=1,
    hemiplegia=2, renal=2, dm_comp=2, cancer=2,
    liver_sev=3,
    mets=6, aids=6,
)

# SQL output filename (case-insensitive on macOS but be defensive).
# Columns are positional (headerless CSV emitted by SSMS):
#   LogID, mi, chf, pvd, cvd, dementia, copd, rheum, pud, liver_mild,
#   dm_uncomp, hemiplegia, renal, dm_comp, cancer, liver_sev, mets, aids,
#   n_ed_visits_180d
LACE_SQL_COLUMNS = [
    "LogID", "mi", "chf", "pvd", "cvd", "dementia", "copd", "rheum",
    "pud", "liver_mild", "dm_uncomp", "hemiplegia", "renal", "dm_comp",
    "cancer", "liver_sev", "mets", "aids", "n_ed_visits_180d",
]
_LACE_DIR = Path("/Users/yiyezhang/Downloads/medhg_ps_data")
_LACE_CANDIDATES = ("LACE_comp.csv", "lace_comp.csv",
                    "lace_components.csv", "LACE_components.csv")
LACE_SQL_OUTPUT = next(
    (_LACE_DIR / n for n in _LACE_CANDIDATES if (_LACE_DIR / n).exists()),
    _LACE_DIR / "LACE_comp.csv",
)


# ---------------------------------------------------------------------
# Component builders
# ---------------------------------------------------------------------
def L_pts(los_days: float) -> int:
    """van Walraven LOS bin scoring."""
    if los_days >= 14: return 7
    if los_days >= 7:  return 5
    if los_days >= 4:  return 4
    if los_days >= 3:  return 3
    if los_days >= 2:  return 2
    if los_days >= 1:  return 1
    return 0


def C_pts_from_sql(row: pd.Series) -> int:
    """Full-Charlson score from the 17 ICD-10-derived flags produced by
    sql/LACE_Components.sql. Cap at 5 (LACE spec)."""
    score = sum(int(row.get(cat, 0)) * w for cat, w in CHARLSON_WEIGHTS.items())
    return min(score, 5)


def C_pts_from_bulk(row: pd.Series) -> int:
    """Partial-Charlson fallback from the WCM comorbidity flags that are
    already in bulk features. Loses MI, CVD, dementia, PVD, COPD, PUD,
    rheumatic, liver categories -- so this systematically underestimates."""
    def yes(v: object) -> bool:
        return str(v).strip().lower() in {"yes", "true", "1",
                                          "insulin", "non-insulin"}
    score = 0
    if yes(row.get("Diabetes Mellitus")):        score += 1
    if yes(row.get("Heart Failure")):            score += 1
    if yes(row.get("Preop Acute Kidney Injury")) \
       or yes(row.get("Preop Dialysis")):        score += 2
    if yes(row.get("Disseminated Cancer")):      score += 6
    if yes(row.get("Immunosuppressive Therapy")): score += 2
    return min(score, 5)


def E_pts(n_ed_180: int) -> int:
    """LACE-E bins: min(count, 4)."""
    return min(int(n_ed_180), 4)


# ---------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------
def compute_lace(
    gold_path: str = "/Users/yiyezhang/Downloads/medhg_ps_data/bulk_features_with_label_gold.parquet",
    emergent_source: str = "auto",          # "ed_first" | "patient_type" | "auto"
) -> pd.DataFrame:
    """Return a DataFrame with LogID + all component points + total LACE_score
    + provenance columns (which components used real vs. proxy data)."""
    merged, feat_cols, cpt_arr, Fseq, seq_all, _ = assemble_training_frame()
    gold = pd.read_parquet(gold_path)[["LogID", "ReadmittedWithin30Days_gold"]]
    merged = merged.assign(LogID=lambda df: df.LogID.astype(str)).merge(
        gold.assign(LogID=lambda df: df.LogID.astype(str)),
        on="LogID", how="left",
    )
    merged["y"] = merged["ReadmittedWithin30Days_gold"].astype(int)

    # ---- L: LOS from A3 ---------------------------------------------
    a3 = d._read_table(C.UNIT_EDGES_PARQUET, C.A3_UNIT_EDGES_COLUMNS).copy()
    a3["LogID"] = a3["LogID"].astype(str)
    a3["InTime"]  = pd.to_datetime(a3["InTime"], errors="coerce")
    a3["OutTime"] = pd.to_datetime(a3["OutTime"], errors="coerce")
    los = a3.groupby("LogID").apply(
        lambda g: (g["OutTime"].max() - g["InTime"].min()).total_seconds() / 86400
    ).rename("los_days").reset_index()
    merged = merged.merge(los, on="LogID", how="left")
    merged["los_days"] = merged["los_days"].fillna(0)
    merged["L_pts"] = merged["los_days"].apply(L_pts)

    # ---- A: emergent flag -------------------------------------------
    first_unit = a3.sort_values("InTime").drop_duplicates("LogID", keep="first")
    first_unit["is_ED_first"] = (
        (first_unit["InstitutionType"] == "ED")
        | (first_unit["UnitType"] == "ED")
    ).astype(int)
    merged = merged.merge(first_unit[["LogID", "is_ED_first"]],
                          on="LogID", how="left")
    merged["is_ED_first"] = merged["is_ED_first"].fillna(0).astype(int)
    merged["patient_type_I"] = (
        merged["PatientType"].astype(str).str.upper() == "I"
    ).astype(int)
    if emergent_source in ("ed_first", "auto"):
        A_source = "ed_first"
        merged["A_pts"] = np.where(merged["is_ED_first"] == 1, 3, 0)
    else:
        A_source = "patient_type"
        merged["A_pts"] = np.where(merged["patient_type_I"] == 1, 3, 0)

    # ---- C: Charlson from SQL if available, partial fallback --------
    if LACE_SQL_OUTPUT.exists():
        # Peek: does row 0 look like a header ("LogID") or a data row?
        first_cell = str(
            pd.read_csv(LACE_SQL_OUTPUT, nrows=1, header=None, dtype=str,
                        encoding="utf-8-sig").iloc[0, 0]
        ).strip()
        has_header = first_cell.upper() == "LOGID"
        lace_sql = pd.read_csv(
            LACE_SQL_OUTPUT,
            header=0 if has_header else None,
            names=None if has_header else LACE_SQL_COLUMNS,
            encoding="utf-8-sig",
            dtype={"LogID": str},
        )
        # Coerce LogID to plain integer-string (drop any float/BOM residue)
        lace_sql["LogID"] = (
            lace_sql["LogID"].astype(str).str.replace(r"\.0+$", "", regex=True)
        )
        for col in list(CHARLSON_WEIGHTS.keys()) + ["n_ed_visits_180d"]:
            if col in lace_sql.columns:
                lace_sql[col] = pd.to_numeric(
                    lace_sql[col], errors="coerce"
                ).fillna(0).astype(int)
        merged = merged.merge(lace_sql, on="LogID", how="left")
        for col in list(CHARLSON_WEIGHTS.keys()) + ["n_ed_visits_180d"]:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0).astype(int)
        merged["C_pts"] = merged.apply(C_pts_from_sql, axis=1)
        merged["E_pts"] = merged["n_ed_visits_180d"].apply(E_pts)
        C_source = "full_icd10"
        E_source = "sql_180d"
    else:
        merged["C_pts"] = merged.apply(C_pts_from_bulk, axis=1)
        C_source = "partial_bulk"
        E_source = "unavailable_zero"
        merged["E_pts"] = 0

    merged["LACE_score"] = (
        merged["L_pts"] + merged["A_pts"] + merged["C_pts"] + merged["E_pts"]
    )
    merged.attrs["A_source"] = A_source
    merged.attrs["C_source"] = C_source
    merged.attrs["E_source"] = E_source
    return merged


def score_to_probability(df: pd.DataFrame) -> np.ndarray:
    """Map integer LACE score -> per-score empirical readmission rate."""
    p = np.zeros(len(df))
    y = df["y"].values
    for s, sub in df.groupby("LACE_score"):
        p[sub.index.values] = float(y[sub.index.values].mean())
    return p


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------
def report(df: pd.DataFrame, tag: str) -> None:
    y = df["y"].values
    p = score_to_probability(df)
    au = roc_auc_score(y, p); ap = average_precision_score(y, p)
    br = brier_score_loss(y, p)
    au_ci = _bootstrap_ci(y, p, roc_auc_score, n_boot=2000, seed=0)
    ap_ci = _bootstrap_ci(y, p, average_precision_score, n_boot=2000, seed=1)

    best_f1, best_thr = 0.0, 0.5
    for t in np.linspace(0.02, 0.35, 100):
        yhat = (p >= t).astype(int)
        if yhat.sum() < 5: continue
        f = f1_score(y, yhat)
        if f > best_f1: best_f1, best_thr = f, t
    yhat = (p >= best_thr).astype(int)

    print(f"\n=== LACE ({tag}) — WCM SQL-fixed, gold label, "
          f"N={len(y):,}, base {y.mean()*100:.2f}% ===")
    print(f"  provenance: A={df.attrs['A_source']}  "
          f"C={df.attrs['C_source']}  E={df.attrs['E_source']}")
    print(f"  component means: L={df['L_pts'].mean():.2f}  "
          f"A={df['A_pts'].mean():.2f}  "
          f"C={df['C_pts'].mean():.2f}  "
          f"E={df['E_pts'].mean():.2f}")
    print(f"  AUROC {au:.3f} ({au_ci[0]:.3f}-{au_ci[1]:.3f})  "
          f"AUPRC {ap:.3f} ({ap_ci[0]:.3f}-{ap_ci[1]:.3f})  Brier {br:.4f}")
    print(f"  Max-F1 thr {best_thr:.3f}: "
          f"F1 {f1_score(y,yhat):.3f}  Prec {precision_score(y,yhat):.3f}  "
          f"Recall {recall_score(y,yhat):.3f}  Flag% {yhat.mean()*100:.1f}")

    # bands
    df = df.copy()
    df["band"] = pd.cut(df["LACE_score"],
                        bins=[-1, 4, 7, 9, 20],
                        labels=["Low (0-4)", "Med (5-7)",
                                "High (8-9)", "Very high (10+)"])
    print(f"  risk stratification:")
    for b, g in df.groupby("band", observed=True):
        print(f"    {str(b):16s}  n={len(g):5d} ({len(g)/len(df)*100:4.1f}%)  "
              f"readmit {g['y'].mean()*100:.2f}%")


def main() -> None:
    print("Loading WCM cohort...")
    have_sql = LACE_SQL_OUTPUT.exists()
    print(f"SQL LACE components file present: {have_sql} ({LACE_SQL_OUTPUT})")
    if not have_sql:
        print("  Falls back to partial Charlson + E=0. Run "
              "sql/LACE_Components.sql to complete the score.")

    df_ed = compute_lace(emergent_source="ed_first")
    report(df_ed, tag="A=ED-first")

    df_pt = compute_lace(emergent_source="patient_type")
    report(df_pt, tag="A=PatientType proxy")

    print(f"\n=== Reference ML baseline on same cohort ===")
    print(f"  rf_clin: AUROC 0.721 (0.706-0.736)  AUPRC 0.172 (0.157-0.192)  Brier 0.066")

    df_ed[[
        "LogID", "los_days", "L_pts", "A_pts", "C_pts", "E_pts",
        "LACE_score", "y", "is_ED_first", "patient_type_I",
    ]].to_csv("artifacts/newdata/lace_wcm_scores.csv", index=False)
    print("\nSaved: artifacts/newdata/lace_wcm_scores.csv")


if __name__ == "__main__":
    main()
