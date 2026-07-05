"""HOSPITAL score (Donzé et al. 2013) clinical-score baseline for
30-day potentially-avoidable readmission on the WCM SQL-fixed cohort.

Components (max total = 13):
    H — Hemoglobin < 12 g/dL at discharge      -> 1 pt
    O — discharge from Oncology service        -> 2 pt
    S — Sodium < 135 mEq/L at discharge        -> 1 pt
    P — Procedure during index admission       -> 1 pt
    I — Index admission type NON-elective      -> 1 pt
    T — Admissions in prior 12 months:
          0-1 -> 0, 2-5 -> 2, >5 -> 5 pt
    A — LOS >= 5 days                          -> 2 pt

Risk categories (Donzé 2013):
    0-4   Low          (~5.8% 30-d readmit in dev cohort)
    5-6   Intermediate (~11.9%)
    >=7   High         (~22.8%)

WCM adaptation and provenance:
    H — HCT < 36 as a proxy for Hb < 12 (Hb ~ HCT/3). PREOP lab in
        our bulk file, not discharge — an approximation. Ideally we
        would pull last-in-encounter Hb from Clarity FLOWSHEETS.
    O — Disseminated Cancer flag from bulk features, used as a
        proxy for oncology-service discharge. Not identical (Donzé
        used the discharge service tag) but the closest available.
    S — NA < 135 from bulk features. Same preop-not-discharge caveat.
    P — Every case in this cohort had surgery -> 1 for all rows.
    I — ED-as-first-unit from A3 (canonical); optional PatientType=I
        fallback (as in analysis/lace_baseline.py). Both are reported
        for transparency.
    T — n_admits_365d from sql/LACE_Components.sql. Falls back to 0
        for every patient if the file lacks that column (i.e. if the
        SQL was run before the HOSPITAL-T extension).
    A — LOS from A3 (max OutTime - min InTime) >= 5 days.

The score is mapped to a probability via per-score empirical readmit
rate, then evaluated with the same protocol as every other model row:
AUROC / AUPRC / Brier with bootstrap CIs, plus F1 / precision / recall
at the max-F1 threshold.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

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

# LACE_Components.sql output (same file used by lace_baseline; new
# HOSPITAL-T column n_admits_365d appended per the SQL update).
_LACE_DIR = Path("/Users/yiyezhang/Downloads/medhg_ps_data")
_LACE_CANDIDATES = ("LACE_comp.csv", "lace_comp.csv",
                    "lace_components.csv", "LACE_components.csv")
LACE_SQL_OUTPUT = next(
    (_LACE_DIR / n for n in _LACE_CANDIDATES if (_LACE_DIR / n).exists()),
    _LACE_DIR / "LACE_comp.csv",
)

# Positional schema of the SQL output (headerless). n_admits_365d is
# absent from files produced before the HOSPITAL-T extension; the
# reader below handles both variants.
LACE_SQL_COLUMNS_LEGACY = [
    "LogID", "mi", "chf", "pvd", "cvd", "dementia", "copd", "rheum",
    "pud", "liver_mild", "dm_uncomp", "hemiplegia", "renal", "dm_comp",
    "cancer", "liver_sev", "mets", "aids", "n_ed_visits_180d",
]
LACE_SQL_COLUMNS_V2 = LACE_SQL_COLUMNS_LEGACY + ["n_admits_365d"]


# ---------------------------------------------------------------------
# Component builders
# ---------------------------------------------------------------------
def _yes(v: object) -> bool:
    return str(v).strip().lower() in {"yes", "true", "1"}


def T_pts(n: int) -> int:
    """HOSPITAL-T binning."""
    if n > 5:  return 5
    if n >= 2: return 2
    return 0


def A_pts(los_days: float) -> int:
    """HOSPITAL 'A' — length of stay >= 5 days -> 2 pts."""
    return 2 if los_days >= 5 else 0


# ---------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------
def compute_hospital_score(
    gold_path: str = "/Users/yiyezhang/Downloads/medhg_ps_data/bulk_features_with_label_gold.parquet",
    emergent_source: Literal["ed_first", "patient_type"] = "ed_first",
) -> pd.DataFrame:
    """Compute the 7-component HOSPITAL score per encounter. Returns a
    DataFrame with LogID + per-component points + total HOSPITAL_score
    + gold outcome, plus provenance attrs."""
    merged, feat_cols, cpt_arr, Fseq, seq_all, _ = assemble_training_frame()
    gold = pd.read_parquet(gold_path)[["LogID", "ReadmittedWithin30Days_gold"]]
    merged = merged.assign(LogID=lambda df: df.LogID.astype(str)).merge(
        gold.assign(LogID=lambda df: df.LogID.astype(str)),
        on="LogID", how="left",
    )
    merged["y"] = merged["ReadmittedWithin30Days_gold"].astype(int)

    # ---- H: Hemoglobin proxy (HCT < 36) -----------------------------
    hct = pd.to_numeric(merged["HCT"], errors="coerce")
    merged["H_pts"] = ((hct < 36) & hct.notna()).astype(int)   # missing -> 0
    H_source = "preop_HCT<36"

    # ---- O: Oncology proxy (Disseminated Cancer flag) ---------------
    merged["O_pts"] = merged["Disseminated Cancer"].apply(_yes).astype(int) * 2
    O_source = "disseminated_cancer_flag"

    # ---- S: Sodium < 135 --------------------------------------------
    na = pd.to_numeric(merged["NA"], errors="coerce")
    merged["S_pts"] = ((na < 135) & na.notna()).astype(int)
    S_source = "preop_Na<135"

    # ---- P: procedure during admission (all patients had surgery) ---
    merged["P_pts"] = 1
    P_source = "all_surgical=1"

    # ---- L (LOS) + A/I from A3 --------------------------------------
    a3 = d._read_table(C.UNIT_EDGES_PARQUET, C.A3_UNIT_EDGES_COLUMNS).copy()
    a3["LogID"] = a3["LogID"].astype(str)
    a3["InTime"]  = pd.to_datetime(a3["InTime"], errors="coerce")
    a3["OutTime"] = pd.to_datetime(a3["OutTime"], errors="coerce")
    los = a3.groupby("LogID").apply(
        lambda g: (g["OutTime"].max() - g["InTime"].min()).total_seconds() / 86400
    ).rename("los_days").reset_index()
    merged = merged.merge(los, on="LogID", how="left")
    merged["los_days"] = merged["los_days"].fillna(0)
    merged["A_pts"] = merged["los_days"].apply(A_pts)
    A_source = "A3_LOS>=5"

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
    if emergent_source == "ed_first":
        merged["I_pts"] = merged["is_ED_first"]
        I_source = "ed_first"
    else:
        merged["I_pts"] = merged["patient_type_I"]
        I_source = "patient_type"

    # ---- T: prior-year admissions from SQL --------------------------
    if LACE_SQL_OUTPUT.exists():
        first_cell = str(
            pd.read_csv(LACE_SQL_OUTPUT, nrows=1, header=None, dtype=str,
                        encoding="utf-8-sig").iloc[0, 0]
        ).strip()
        has_header = first_cell.upper() == "LOGID"
        # Detect schema by column count
        ncols = pd.read_csv(LACE_SQL_OUTPUT, nrows=1, header=None,
                            encoding="utf-8-sig").shape[1]
        schema = LACE_SQL_COLUMNS_V2 if ncols >= 20 else LACE_SQL_COLUMNS_LEGACY
        lace_sql = pd.read_csv(
            LACE_SQL_OUTPUT,
            header=0 if has_header else None,
            names=None if has_header else schema,
            encoding="utf-8-sig",
            dtype={"LogID": str},
        )
        lace_sql["LogID"] = (
            lace_sql["LogID"].astype(str).str.replace(r"\.0+$", "", regex=True)
        )
        has_T = "n_admits_365d" in lace_sql.columns
        if has_T:
            lace_sql["n_admits_365d"] = pd.to_numeric(
                lace_sql["n_admits_365d"], errors="coerce"
            ).fillna(0).astype(int)
            merged = merged.merge(
                lace_sql[["LogID", "n_admits_365d"]], on="LogID", how="left"
            )
            merged["n_admits_365d"] = merged["n_admits_365d"].fillna(0).astype(int)
            merged["T_pts"] = merged["n_admits_365d"].apply(T_pts)
            T_source = "sql_365d"
        else:
            merged["n_admits_365d"] = 0
            merged["T_pts"] = 0
            T_source = "sql_missing_column_zero"
    else:
        merged["n_admits_365d"] = 0
        merged["T_pts"] = 0
        T_source = "unavailable_zero"

    merged["HOSPITAL_score"] = (
        merged["H_pts"] + merged["O_pts"] + merged["S_pts"] + merged["P_pts"]
        + merged["I_pts"] + merged["T_pts"] + merged["A_pts"]
    )
    merged.attrs = dict(
        H_source=H_source, O_source=O_source, S_source=S_source,
        P_source=P_source, I_source=I_source, T_source=T_source,
        A_source=A_source,
    )
    return merged


def score_to_probability(df: pd.DataFrame) -> np.ndarray:
    """Empirical readmit rate per score bin."""
    p = np.zeros(len(df))
    y = df["y"].values
    for s, sub in df.groupby("HOSPITAL_score"):
        p[sub.index.values] = float(y[sub.index.values].mean())
    return p


def _report(df: pd.DataFrame, tag: str) -> None:
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

    print(f"\n=== HOSPITAL ({tag}) — WCM SQL-fixed, gold label, "
          f"N={len(y):,}, base {y.mean()*100:.2f}% ===")
    print("  provenance:")
    for k, v in df.attrs.items():
        print(f"    {k:10s} {v}")
    print(f"  component means: "
          f"H={df['H_pts'].mean():.2f}  "
          f"O={df['O_pts'].mean():.2f}  "
          f"S={df['S_pts'].mean():.2f}  "
          f"P={df['P_pts'].mean():.2f}  "
          f"I={df['I_pts'].mean():.2f}  "
          f"T={df['T_pts'].mean():.2f}  "
          f"A={df['A_pts'].mean():.2f}")
    print(f"  AUROC {au:.3f} ({au_ci[0]:.3f}-{au_ci[1]:.3f})   "
          f"AUPRC {ap:.3f} ({ap_ci[0]:.3f}-{ap_ci[1]:.3f})   Brier {br:.4f}")
    print(f"  Max-F1 thr {best_thr:.3f}: "
          f"F1 {f1_score(y,yhat):.3f}  Prec {precision_score(y,yhat):.3f}  "
          f"Recall {recall_score(y,yhat):.3f}  Flag% {yhat.mean()*100:.1f}")

    df = df.copy()
    df["band"] = pd.cut(
        df["HOSPITAL_score"],
        bins=[-1, 4, 6, 20],
        labels=["Low (0-4)", "Intermediate (5-6)", "High (>=7)"],
    )
    print("  risk stratification:")
    for b, g in df.groupby("band", observed=True):
        print(f"    {str(b):22s}  n={len(g):5d} ({len(g)/len(df)*100:4.1f}%)  "
              f"readmit {g['y'].mean()*100:.2f}%")


def main() -> None:
    print("Loading WCM cohort...")
    print(f"SQL LACE/HOSPITAL components file: {LACE_SQL_OUTPUT} "
          f"(exists={LACE_SQL_OUTPUT.exists()})")

    df_ed = compute_hospital_score(emergent_source="ed_first")
    _report(df_ed, tag="I=ED-first (canonical)")

    df_pt = compute_hospital_score(emergent_source="patient_type")
    _report(df_pt, tag="I=PatientType proxy")

    print("\n=== Reference ML baseline on same cohort ===")
    print("  rf_clin: AUROC 0.721 (0.706-0.736)  AUPRC 0.172 (0.157-0.192)  Brier 0.066")
    print("  LACE (A=ED-first, C+E from SQL): AUROC 0.679  AUPRC 0.139  F1 0.225")

    Path("artifacts/newdata").mkdir(parents=True, exist_ok=True)
    df_ed[[
        "LogID", "los_days", "H_pts", "O_pts", "S_pts", "P_pts",
        "I_pts", "T_pts", "A_pts", "HOSPITAL_score", "y",
        "is_ED_first", "patient_type_I", "n_admits_365d",
    ]].to_csv("artifacts/newdata/hospital_wcm_scores.csv", index=False)
    print("\nSaved: artifacts/newdata/hospital_wcm_scores.csv")


if __name__ == "__main__":
    main()
