"""
Configuration and hyperparameter search space.

Reproduces the search ranges from Chen et al. (npj Health Systems 2025),
Table 2 row "MedHG-PS" exactly. Defaults are the values reported in the
paper's text where stated, otherwise the midpoint of the range.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


def _resolve_device() -> str:
    """Pick the compute device: MEDHG_PS_DEVICE env override wins; else
    cuda if a CUDA build+GPU is present; else cpu."""
    forced = os.environ.get("MEDHG_PS_DEVICE")
    if forced:
        return forced
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
# All paths can be overridden by environment variable so the same
# config works across the user's Windows box (where Cogito extracts run),
# Mac dev box (where GNN training runs), and Cogito Cloud without code
# edits. Default DATA_DIR is ~/Downloads/medhg_ps_data so the CSVs
# saved out of Cogito (Windows or Mac) land where the pipeline reads
# them with zero env-var setup.
PROJECT_ROOT = Path(os.environ.get(
    "MEDHG_PS_PROJECT_ROOT",
    str(Path(__file__).resolve().parent.parent),
))
DATA_DIR     = Path(os.environ.get(
    "MEDHG_PS_DATA_DIR",
    str(Path.home() / "Downloads" / "medhg_ps_data"),
))
ARTIFACT_DIR = Path(os.environ.get("MEDHG_PS_ARTIFACT_DIR",
                                   str(PROJECT_ROOT / "medhg_ps" / "artifacts")))
# Embeddings are bulky model OUTPUTS, not source code -- keep them out of
# the Dropbox-synced repo (PROJECT_ROOT) and alongside the data inputs in
# DATA_DIR instead. Override with MEDHG_PS_EMBED_DIR if needed.
EMBED_DIR    = Path(os.environ.get("MEDHG_PS_EMBED_DIR",
                                   str(DATA_DIR / "embeddings")))

# Parquet outputs from GNN_Graph_Extract.sql Part A.
ENCOUNTERS_PARQUET = Path(os.environ.get("MEDHG_PS_ENCOUNTERS",
                                         str(DATA_DIR / "A1_encounters.parquet")))
PROV_EDGES_PARQUET = Path(os.environ.get("MEDHG_PS_PROV_EDGES",
                                         str(DATA_DIR / "A2_enc_prov_edges.parquet")))
UNIT_EDGES_PARQUET = Path(os.environ.get("MEDHG_PS_UNIT_EDGES",
                                         str(DATA_DIR / "A3_enc_unit_edges.parquet")))
PROV_ATTRS_PARQUET = Path(os.environ.get("MEDHG_PS_PROV_ATTRS",
                                         str(DATA_DIR / "A4_prov_attrs.parquet")))
UNIT_ATTRS_PARQUET = Path(os.environ.get("MEDHG_PS_UNIT_ATTRS",
                                         str(DATA_DIR / "A5_unit_attrs.parquet")))

# Order-sequence substrate (Order_Sequence_Extract.sql) -- replaces the
# care-unit sequence (A3/A5) as the GRU/GNN sequence. Units dropped;
# providers (A2/A4) retained.
ORDER_SEQ_PARQUET  = Path(os.environ.get("MEDHG_PS_ORDER_SEQ",
                                         str(DATA_DIR / "order_sequence.parquet")))

# Tabular encounter features + 30-day readmission label.
#
# Originally Surgery_RVA.csv (NSQIP-abstracted, 6018 cases) -- now
# bulk_features_with_label.parquet, produced by running
# Bulk_Features_From_Cohort.sql against the 43k Clarity-derived MIS
# cohort. The parquet contains both the 40 model features and the
# ReadmittedWithin30Days label, so ENC_FEATURES_CSV and LABELS_PARQUET
# default to the same file. data.load_raw() detects parquet vs csv by
# extension via _read_table().
#
# Defaults assume the file is at ~/Downloads/medhg_ps_data/
# bulk_features_with_label.parquet (same DATA_DIR as the A1..A5
# parquets). Override either var to point elsewhere.
ENC_FEATURES_CSV   = Path(os.environ.get(
    "MEDHG_PS_ENC_FEATURES_CSV",
    str(DATA_DIR / "bulk_features_with_label.parquet"),
))

# Map: Surgery_RVA.csv (raw NSQIP-abstraction) column name -> the
# column name PeriOp_NSQIP_model.py produces from Epic Clarity. The
# right-hand side is the canonical inference-time schema (40 model
# features in artifacts['fill_values']) and MUST match the keys
# populated in PeriOp_NSQIP_model.py:_build_per_case_rows so the
# trained pickle is interchangeable between training and serving.
#
# Columns from the raw CSV that are NOT in this dict are dropped by
# the model (e.g. CPT Description, Operative Approach, all # of
# Postop outcomes other than the three intraop/PACU ones we keep,
# Functional Health Status (dropped per [[nsqip-comorbidity-derivation-fidelity]]),
# Hemoglobin/HbA1c, Surgical/Hospital LOS, Postoperative ICD10).
NSQIP_TO_SQL_RENAMES = {
    # ---- demographics / case ----
    "Age at Time of Surgery":                          "AgeYears",
    "Sex":                                             "Gender",
    "In/Out-Patient Status":                           "PatientType",
    "ASA Classification":                              "ASAClass",
    "Principal Anesthesia Technique":                  "AnesType",
    "Hospital Discharge Destination":                  "Discharge Disposition",
    "Height":                                          "Height (cm)",
    "Weight":                                          "Weight (kg)",
    "Duration of Surgical Procedure (in minutes)":     "CutToClose",
    # ---- procedure counts (same name in both schemas) ----
    "# of Other Procedures":                           "# of Other Procedures",
    "# of Concurrent Procedures":                      "# of Concurrent Procedures",
    # ---- labs ----
    "Serum Sodium":                                    "NA",
    "BUN":                                             "BUN",
    "Serum Creatinine":                                "Creat",
    "Albumin":                                         "ALB",
    "Total Bilirubin":                                 "BT",
    "AST/SGOT":                                        "SGOT",
    "Alkaline Phosphatase":                            "ALKPhos",
    "WBC":                                             "WBC",
    "Hematocrit":                                      "HCT",
    "Platelet Count":                                  "PLT",
    "INR":                                             "INR",
    "PTT":                                             "APTT",
    # ---- comorbidities (CSV already uses target names) ----
    "Diabetes Mellitus":                               "Diabetes Mellitus",
    "Current Smoker within 1 year":                    "Current Smoker within 1 year",
    "Ventilator Dependent":                            "Ventilator Dependent",
    "History of Severe COPD":                          "History of Severe COPD",
    "Ascites":                                         "Ascites",
    "Heart Failure":                                   "Heart Failure",
    "Hypertension requiring medication":               "Hypertension requiring medication",
    "Preop Acute Kidney Injury":                       "Preop Acute Kidney Injury",
    "Preop Dialysis":                                  "Preop Dialysis",
    "Disseminated Cancer":                             "Disseminated Cancer",
    "Immunosuppressive Therapy":                       "Immunosuppressive Therapy",
    "Bleeding Disorder":                               "Bleeding Disorder",
    "Preop RBC Transfusions (72h)":                    "Preop RBC Transfusions (72h)",
    # ---- intraop / PACU events kept as features (model.timing memo) ----
    "# of Cardiac Arrest Requiring CPR":               "# of Cardiac Arrest Requiring CPR",
    "# of Stroke/Cerebral Vascular Acccident (CVA)":   "# of Stroke/Cerebral Vascular Acccident (CVA)",
    "# of Postop Unplanned Intubation":                "# of Postop Unplanned Intubation",
}

# Explicit allow-list of the model features (matches the keys populated
# in PeriOp_NSQIP_model.py:_build_per_case_rows). Anything not in this
# set is either a label, an ID, or a non-Epic-derivable feature and must
# be excluded from training inputs.
#
# DROPPED: "Height (cm)" and "BMI". PAT_ENC.HEIGHT does not parse to
# decimal inches on this Clarity build (TRY_CONVERT -> NULL for every
# row), so Height is 100% null and BMI (which needs it) collapses too.
# Weight (kg) is fine (~96% populated) and is retained. -> 38 features.
# PrimaryCPT (top-RVU MIS procedure code) and the SDoH block are appended
# as model features. They are not NSQIP renames, so added explicitly here.
# EXCLUDED as predictors (kept in the bulk file for other uses):
#   * ZIP                  -- high-cardinality; for external ADI/SVI linkage only.
#   * Race, Ethnicity      -- equity hazard as model inputs (can encode bias);
#                             retained for STRATIFIED EVALUATION of fairness.
# (35 NSQIP) + 1 PrimaryCPT + 5 SDoH (Language, 4 Z-flags) = 41.
# Dropped as low-quality vs the NSQIP gold standard (poor concordance; removing
# them slightly improved AUROC/AUPRC): Bleeding Disorder and History of Severe
# COPD (both under-capture true cases) and # of Other Procedures (count mismatch).
_DROPPED_LOW_QUALITY = ("Bleeding Disorder", "History of Severe COPD", "# of Other Procedures")
MODEL_FEATURE_COLUMNS = tuple(
    v for v in NSQIP_TO_SQL_RENAMES.values()
    if v != "Height (cm)" and v not in _DROPPED_LOW_QUALITY
) + (
    "PrimaryCPT",
    "Language",
    "SDOH_Housing_Z", "SDOH_Food_Z", "SDOH_Financial_Z", "SDOH_Any_Z",
)
assert len(set(MODEL_FEATURE_COLUMNS)) == 41, (
    f"Expected 41 model features, got {len(set(MODEL_FEATURE_COLUMNS))}"
)

# Authoritative column schema for bulk_features_with_label, in the EXACT
# order emitted by the final SELECT of Bulk_Features_From_Cohort.sql
# (5 identifiers + 40 model features + 2 label columns + PrimaryCPT = 48).
# PrimaryCPT is appended LAST by the SQL (after the label) so the original
# column order is unchanged.
#
# This list IS the schema contract: data._read_bulk_features never trusts
# the on-disk header (Cogito exports the file with or without one), it
# applies these names positionally instead. So the export needs no header
# at all -- but keep this list byte-for-byte in sync with the SQL SELECT,
# because a drift here silently mislabels every column.
BULK_FEATURES_COLUMNS = (
    # ---- identifiers ----
    "LogID", "PAT_ID", "EncounterCSN", "SurgeryDate", "PAT_MRN_ID",
    # ---- 40 model features (SELECT order; BMI sits inline after Weight) ----
    "AgeYears", "Gender", "PatientType", "ASAClass", "AnesType",
    "Discharge Disposition", "Height (cm)", "Weight (kg)", "BMI",
    "CutToClose", "# of Other Procedures", "# of Concurrent Procedures",
    "NA", "BUN", "Creat", "ALB", "BT", "SGOT", "ALKPhos", "WBC", "HCT",
    "PLT", "INR", "APTT",
    "Diabetes Mellitus", "Current Smoker within 1 year",
    "Ventilator Dependent", "History of Severe COPD", "Ascites",
    "Heart Failure", "Hypertension requiring medication",
    "Preop Acute Kidney Injury", "Preop Dialysis", "Disseminated Cancer",
    "Immunosuppressive Therapy", "Bleeding Disorder",
    "Preop RBC Transfusions (72h)", "# of Cardiac Arrest Requiring CPR",
    "# of Stroke/Cerebral Vascular Acccident (CVA)",
    "# of Postop Unplanned Intubation",
    # ---- label ----
    "DaysToReadmission", "ReadmittedWithin30Days",
    # ---- procedure code (appended after the label by the SQL) ----
    "PrimaryCPT",
    # ---- SDoH block (appended after PrimaryCPT; ORDER MUST MATCH the
    #      final SELECT in Bulk_Features_From_Cohort.sql) ----
    "Race", "Ethnicity", "Language", "ZIP",
    "SDOH_Housing_Z", "SDOH_Food_Z", "SDOH_Financial_Z", "SDOH_Any_Z",
)
assert len(BULK_FEATURES_COLUMNS) == 56, (
    f"Expected 56 bulk-feature columns, got {len(BULK_FEATURES_COLUMNS)}"
)

# ---------------------------------------------------------------------
# A1..A5 graph-extract schemas (the single source of truth: the final
# SELECT of each section in GNN_Graph_Extract.sql, in emitted order).
#
# Like BULK_FEATURES_COLUMNS, these let data._read_table apply column
# names POSITIONALLY so the extracts need no header on disk. The latest
# Cogito batch exports every table headerless; keep each tuple
# byte-for-byte in sync with GNN_Graph_Extract.sql or the columns get
# silently mislabeled. (CSV is lossless when headerless; a headerless
# PARQUET still drops its first data row -- see data._read_with_schema.)
# ---------------------------------------------------------------------
A1_ENCOUNTERS_COLUMNS = (
    "LogID", "EncounterCSN", "PAT_ID", "SurgeryDate", "PrimarySurgID",
)
A2_PROV_EDGES_COLUMNS = (
    "LogID", "ProvID", "RoleCode", "RoleSource", "ProviderType",
)
A3_UNIT_EDGES_COLUMNS = (
    "LogID", "EncounterCSN", "PAT_ID", "DepartmentID", "DepartmentName",
    "UnitType", "InstitutionType", "InTime", "OutTime", "Hours",
    "SeqInEncounter",
)
A4_PROV_ATTRS_COLUMNS = (
    "ProvID", "ProvName", "EmployedCRNA", "IsResident", "IsHospitalist",
    "DoctorsDegree", "ClinicianTitle", "ProvType", "StaffResourceCode",
    "MCDProfCode", "CaseVolume2yr", "CaseVolume5yr",
)
A5_UNIT_ATTRS_COLUMNS = (
    "DepartmentID", "DepartmentName", "Specialty", "InpatientFlag",
    "ADTUnitTypeCode", "ORUnitTypeCode", "IsPeriopDept", "LicensedBeds",
    "ServiceAreaID", "UnitType", "InstitutionType",
)
# Order-sequence extract (Order_Sequence_Extract.sql final SELECT order).
# Raw stream; runs are collapsed at load time (data.collapse_order_runs).
ORDER_SEQ_COLUMNS = (
    "LogID", "PAT_ID", "EncounterCSN", "SeqInEncounter",
    "OrderTime", "OrderSource", "OrderGroup", "MinutesFromPrev",
)

# Encounter-level care-team block (Option B) from A2 edges + A4 attrs.
# Built by data.build_provider_team_features(); a tabular block, no per-order
# attribution. Surgeon volume from A1.PrimarySurgID -> A4.CaseVolume*.
PROVIDER_FEATURE_COLUMNS = (
    "team_size", "n_residents", "n_crna", "n_hospitalists",
    "has_resident", "has_crna", "has_hospitalist",
    "surg_casevol_2yr", "surg_casevol_5yr",
    # 1 when PrimarySurgID didn't match A4 (no surgeon-volume data) -- a
    # distinct population from low-volume (readmits ~10.7% vs Q1 8.3%),
    # so the tree can split on it rather than treating 0 as low-volume.
    "surg_vol_unmatched",
)
for _name, _cols, _n in (
    ("A1_ENCOUNTERS_COLUMNS", A1_ENCOUNTERS_COLUMNS, 5),
    ("A2_PROV_EDGES_COLUMNS", A2_PROV_EDGES_COLUMNS, 5),
    ("A3_UNIT_EDGES_COLUMNS", A3_UNIT_EDGES_COLUMNS, 11),
    ("A4_PROV_ATTRS_COLUMNS", A4_PROV_ATTRS_COLUMNS, 12),
    ("A5_UNIT_ATTRS_COLUMNS", A5_UNIT_ATTRS_COLUMNS, 11),
    ("ORDER_SEQ_COLUMNS",     ORDER_SEQ_COLUMNS,      8),
):
    assert len(_cols) == _n, f"{_name}: expected {_n} cols, got {len(_cols)}"

# Encounter-level features DERIVED from the graph extracts, per the
# MedHG-PS paper's H^ENC definition (Chen et al. npj Health Systems
# 2025, Methods): the encounter vector is the NSQIP tabular block PLUS
# a "transfer history" block + calendar block.
#
# Pre-operative care-unit trajectory from A3 (split at surgery start so
# it stays leakage-safe -- only ward time BEFORE the index operation):
#   preop_los_{acute,intensive,intermediate}_hr  - hours in each ward type
#   preop_transfer_count                         - # pre-op ward visits
#   preop_n_units                                - # distinct pre-op depts
TRAJECTORY_FEATURE_COLUMNS = (
    "preop_los_acute_hr",
    "preop_los_intensive_hr",
    "preop_los_intermediate_hr",
    "preop_transfer_count",
    "preop_n_units",
)
# Calendar block from A1 SurgeryDate / surgery start timestamp.
CALENDAR_FEATURE_COLUMNS = (
    "surgery_dow",          # day-of-week name (one-hot)
    "surgery_is_weekend",   # 0/1
)
# Full encounter feature set = NSQIP allow-list + derived blocks.
ENCOUNTER_DERIVED_COLUMNS = TRAJECTORY_FEATURE_COLUMNS + CALENDAR_FEATURE_COLUMNS

# 30-day readmission label (one row per LogID). Same file as
# ENC_FEATURES_CSV by default -- the bulk-features parquet carries the
# ReadmittedWithin30Days column. Override if you want labels stored
# separately.
LABELS_PARQUET     = Path(os.environ.get(
    "MEDHG_PS_LABELS",
    str(DATA_DIR / "bulk_features_with_label.parquet"),
))


# ---------------------------------------------------------------------
# Graph schema
# ---------------------------------------------------------------------
# Node types - paper's ENC / P / C with provider subtypes used by ablation.
ENC_NTYPE  = "encounter"
PROV_NTYPE = "provider"
UNIT_NTYPE = "unit"
NODE_TYPES = (ENC_NTYPE, PROV_NTYPE, UNIT_NTYPE)

# Edge types - bidirectional because the paper's graph is undirected.
# DGL canonical edges are (src_ntype, etype, dst_ntype) triples.
ETYPE_ENC_PROV = (ENC_NTYPE,  "has_provider",    PROV_NTYPE)
ETYPE_PROV_ENC = (PROV_NTYPE, "treated",         ENC_NTYPE)
ETYPE_ENC_UNIT = (ENC_NTYPE,  "visited_unit",    UNIT_NTYPE)
ETYPE_UNIT_ENC = (UNIT_NTYPE, "hosted",          ENC_NTYPE)
EDGE_TYPES = (ETYPE_ENC_PROV, ETYPE_PROV_ENC, ETYPE_ENC_UNIT, ETYPE_UNIT_ENC)

# Provider subtypes (matches paper Methods section).
PROVIDER_SUBTYPES = (
    "SurgicalTeam",     # surgeons, assistants, residents, fellows
    "OtherClinician",   # anesthesiologists, CRNAs, PAs
    "Nurse",            # circulating, scrub, PACU, ICU
    "Technician",       # perfusionists, scrub techs, etc.
    "Other",            # observers, ancillary
)

# Care-unit subtypes (paper's three subgroups).
UNIT_SUBTYPES = ("Intensive", "Intermediate", "Acute")


# ---------------------------------------------------------------------
# Hyperparameters - default values
# ---------------------------------------------------------------------
@dataclass
class TrainConfig:
    # Optimizer (paper: Adam, lr/L2 reg in [1e-7, 1e-1], TPE-tuned).
    learning_rate: float = 1e-3
    l2_reg: float        = 1e-4
    optimizer: str       = "adam"

    # Regularization (paper: dropout [0.0, 0.8], batch_norm True/False).
    dropout: float       = 0.3
    batch_norm: bool     = True

    # Architecture (paper: first hidden dim in {32,64,128,256}, attention
    # vector dim in {8,16,32,64,128}, three GCN layers).
    hidden_dim_1: int    = 128
    hidden_dim_2: int    = 64    # paper doesn't specify; halve per layer
    hidden_dim_3: int    = 32
    attn_dim: int        = 32
    n_layers: int        = 3

    # Training loop (paper: batch size 512, BCE, early stopping on val AUROC).
    batch_size: int      = 512
    max_epochs: int      = 200
    early_stop_patience: int = 20
    eval_every: int      = 1

    # Resampling for class imbalance (paper compared undersampling,
    # oversampling, and SMOTE; defaults to no resampling here).
    resampling: str      = "none"  # one of: none / undersample / oversample / smote

    # Data split (paper: 8:1:1 train/val/test).
    train_frac: float    = 0.8
    val_frac: float      = 0.1
    test_frac: float     = 0.1
    split_seed: int      = 42

    # Device. Resolved at construction: honor MEDHG_PS_DEVICE if set,
    # else use cuda when actually available, otherwise fall back to cpu
    # (so Mac/CPU-only dev boxes work without a code edit).
    device: str          = field(default_factory=lambda: _resolve_device())


@dataclass
class HPSearchSpace:
    """Bayesian / TPE search ranges - matches paper Table 2 exactly."""
    learning_rate: Tuple[float, float] = (1e-7, 1e-1)        # log-uniform
    l2_reg:        Tuple[float, float] = (1e-7, 1e-1)        # log-uniform
    dropout:       Tuple[float, float] = (0.0, 0.8)          # uniform
    batch_norm:    Tuple[bool, bool]   = (False, True)       # categorical
    hidden_dim_1:  Tuple[int, ...]     = (32, 64, 128, 256)  # categorical
    attn_dim:      Tuple[int, ...]     = (8, 16, 32, 64, 128)  # categorical
    n_calls: int                       = 50                  # TPE iterations


@dataclass
class FeaturePrepConfig:
    """Feature preprocessing knobs - matches paper Methods § Data preprocessing."""
    # Outlier filtering (paper: "filtering out outliers"; concrete rule not
    # given - we use IQR-1.5 fence on continuous columns and clip rather
    # than drop so we keep the encounter for graph connectivity).
    outlier_method: str  = "iqr_clip"
    iqr_multiplier: float = 1.5

    # Imputation (paper: "Unknown" for categorical, mean for continuous).
    categorical_fill: str = "Unknown"
    continuous_fill:  str = "mean"

    # One-hot (paper: applied to all categorical variables).
    drop_first_in_ohe: bool = False  # paper does not drop; keep all columns

    # Standardization (paper: "standardizing continuous variables"; we use
    # StandardScaler, fit on train only).
    standardize: bool = True

    # Random masking experiment (paper: BMI 4.3%, height 4.3%, CCI 8.3%,
    # marital 2.11%, ASA-PS 1.7%). Off for training, enabled for ablations.
    enable_random_masking: bool = False
    masking_rates: dict = field(default_factory=lambda: {
        "BMI":            0.043,
        "Height (cm)":    0.043,
        "ASAClass":       0.017,
        "MaritalStatus":  0.0211,
        "CCI":            0.083,
    })


# Singleton defaults that the rest of the package imports.
DEFAULTS_TRAIN   = TrainConfig()
DEFAULTS_SEARCH  = HPSearchSpace()
DEFAULTS_FEATS   = FeaturePrepConfig()
