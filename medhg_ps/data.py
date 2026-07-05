"""
Data loading and feature preprocessing.

Reproduces the preprocessing pipeline described in Chen et al. (npj
Health Systems 2025) Methods § Data preprocessing:

    "Before loading inputs to the framework, the raw data was preprocessed
     by filtering out outliers, standardizing continuous variables, and
     applying one-hot encoding to categorical variables. Missing data were
     addressed by assigning 'Unknown' to missing categorical variables and
     imputing missing continuous variables using their mean values."

Order:
    1. Load A1..A5 parquet outputs + tabular encounter features + labels.
    2. Filter outliers (IQR-1.5 clip on continuous columns).
    3. Impute (categorical -> "Unknown", continuous -> training-set mean).
    4. (Optional) random masking experiment.
    5. One-hot encode categorical columns.
    6. Standardize continuous columns with StandardScaler fitted on train.

All scalers/encoders are fit on the train split only and stored in a
PreprocessState dataclass so they can be re-applied to val/test/inference.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import config as C


# ---------------------------------------------------------------------
# Raw load
# ---------------------------------------------------------------------
@dataclass
class RawExtract:
    """The five tables emitted by GNN_Graph_Extract.sql Part A + the
    NSQIP tabular features + the 30-day readmission label."""
    encounters:    pd.DataFrame   # A1
    enc_prov_edges: pd.DataFrame  # A2
    enc_unit_edges: pd.DataFrame  # A3
    prov_attrs:    pd.DataFrame   # A4
    unit_attrs:    pd.DataFrame   # A5
    enc_features:  pd.DataFrame   # Surgery_RVA tabular, joined on LogID
    labels:        pd.DataFrame   # LogID -> ReadmittedWithin30Days {0,1}


def _apply_nsqip_renames(
    df: pd.DataFrame,
    rename_map: Dict[str, str] = C.NSQIP_TO_SQL_RENAMES,
) -> pd.DataFrame:
    """Rename NSQIP-Operations-Manual columns to the SQL CTE column
    names, matching the in-notebook renaming in cell 6b6c41d9 of
    Surgery_shortmodel.ipynb. Unknown source columns are left alone so
    that an already-renamed CSV is also accepted."""
    applicable = {k: v for k, v in rename_map.items() if k in df.columns}
    return df.rename(columns=applicable)


def load_raw(
    encounters_path: Path = C.ENCOUNTERS_PARQUET,
    prov_edges_path: Path = C.PROV_EDGES_PARQUET,
    unit_edges_path: Path = C.UNIT_EDGES_PARQUET,
    prov_attrs_path: Path = C.PROV_ATTRS_PARQUET,
    unit_attrs_path: Path = C.UNIT_ATTRS_PARQUET,
    enc_features_path: Path = C.ENC_FEATURES_CSV,
    labels_path: Path = C.LABELS_PARQUET,
    rename_map: Optional[Dict[str, str]] = None,
) -> RawExtract:
    enc_feat = _read_table(enc_features_path, schema=C.BULK_FEATURES_COLUMNS)
    enc_feat = _apply_nsqip_renames(
        enc_feat,
        rename_map=rename_map if rename_map is not None else C.NSQIP_TO_SQL_RENAMES,
    )
    # Units (A3/A5) are optional: the order-sequence track drops them, so
    # tolerate missing files by returning empty frames (unit-based scripts
    # then simply see no trajectory; the order/provider track ignores them).
    def _read_opt(path, schema):
        return (_read_table(path, schema=schema) if Path(path).exists()
                else pd.DataFrame(columns=list(schema)))

    return RawExtract(
        encounters    = _read_table(encounters_path, schema=C.A1_ENCOUNTERS_COLUMNS),
        enc_prov_edges = _read_table(prov_edges_path, schema=C.A2_PROV_EDGES_COLUMNS),
        enc_unit_edges = _read_opt(unit_edges_path, C.A3_UNIT_EDGES_COLUMNS),
        prov_attrs    = _read_table(prov_attrs_path, schema=C.A4_PROV_ATTRS_COLUMNS),
        unit_attrs    = _read_opt(unit_attrs_path, C.A5_UNIT_ATTRS_COLUMNS),
        enc_features  = enc_feat,
        labels        = _read_table(labels_path, schema=C.BULK_FEATURES_COLUMNS),
    )


# Columns that are identifiers, never numeric. SQL emits them via
# CONVERT(varchar, ...) so they are strings on the DB side, but
# pandas re-infers numeric dtype when reading headerless CSVs (whose
# values look like big integers). That dtype-mismatch breaks merges
# downstream (e.g. merged_all.LogID int64 vs traj.LogID str).
# Normalize at the read boundary so the rest of the pipeline never
# has to think about it.
_ID_COLUMNS_TO_STRING = (
    "LogID", "EncounterCSN", "PAT_ID",
    "ProvID", "PrimarySurgID",
    "DepartmentID",
)


def _read_with_schema(path: Path, schema: Sequence[str]) -> pd.DataFrame:
    """Read a graph extract / bulk-features table header-agnostically.

    Each table's column order is fixed by its final SELECT in
    GNN_Graph_Extract.sql / Bulk_Features_From_Cohort.sql, mirrored by the
    matching *_COLUMNS tuple in config -- that tuple, not the file, is the
    source of truth. The latest Cogito batch exports every table WITHOUT a
    header, so we never trust the on-disk header: we detect whether row 0
    is a header or data, read accordingly, then assign `schema`
    POSITIONALLY.

    Headerless handling differs by format:
      * CSV  -- pandas' default header=0 would eat data-row 0 as names; we
        read with header=None so every row is preserved (CSV is lossless).
      * Parquet -- a headerless write puts row-0's values INTO the schema
        field names, where pandas has already mangled them (duplicate
        values gain .1/.2/... suffixes; dates/numerics are type-coerced).
        That row cannot be faithfully recovered -- e.g. a row whose
        0-valued columns became names 0, 0.1, ... 0.5 would inject the
        literal '0.5' into a binary column and blow up at astype(int). We
        drop that single row; re-export the parquet WITH a header to keep
        it. (CSV has no such limitation -- prefer CSV when headerless.)
    """
    expected = list(schema)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        # Peek at the first cell to tell a header row from a data row.
        first_cell = str(pd.read_csv(path, nrows=1, header=None,
                                     dtype=str).iloc[0, 0]).strip()
        has_header = first_cell == expected[0]
        df = pd.read_csv(path, header=0 if has_header else None,
                         low_memory=False)
    elif suffix in (".parquet", ".pq"):
        df = pd.read_parquet(path)
        # Headerless parquet -> row-0 is gone (consumed into field names).
        # Nothing to re-read; we proceed with the surviving rows.
    else:
        raise ValueError(
            f"Unsupported file extension {suffix} for {path}. "
            "Expected .parquet, .pq, or .csv."
        )

    if df.shape[1] != len(expected):
        raise ValueError(
            f"{path.name}: expected {len(expected)} columns "
            f"(per the SQL extract), got {df.shape[1]}. "
            "Header row may be missing/extra or the SQL SELECT changed."
        )
    df.columns = expected           # SQL SELECT order is authoritative
    return df


def _read_table(path: Path,
                schema: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Read a table from parquet OR csv based on extension. Makes the
    pipeline agnostic to whichever format the user exports from Cogito.

    When `schema` is given (every load_raw table supplies one), the read
    is header-agnostic: column names come from the SQL-derived schema, not
    the file, so the export needs no header. Coerces known ID columns to
    str so cross-source merges line up."""
    if schema is not None:
        df = _read_with_schema(path, schema)
    else:
        suffix = path.suffix.lower()
        if suffix in (".parquet", ".pq"):
            df = pd.read_parquet(path)
        elif suffix == ".csv":
            df = pd.read_csv(path, low_memory=False)
        else:
            raise ValueError(
                f"Unsupported file extension {suffix} for {path}. "
                "Expected .parquet, .pq, or .csv."
            )
    for col in _ID_COLUMNS_TO_STRING:
        if col in df.columns:
            # Trim the '.0' that pandas float coercion leaves on numeric
            # IDs that got read as float (e.g. when a column has NaNs).
            df[col] = (df[col].astype(str).str.strip()
                       .str.replace(r"\.0+$", "", regex=True))
    return df


# ---------------------------------------------------------------------
# Order-sequence substrate (Order_Sequence_Extract.sql) -- the GRU/GNN
# sequence, replacing the care-unit path (A3). Providers (A2/A4) unaffected.
# ---------------------------------------------------------------------
def load_order_sequence(path: Path = C.ORDER_SEQ_PARQUET) -> pd.DataFrame:
    """Raw per-order stream, positional schema applied. One row per order."""
    df = _read_table(path, schema=list(C.ORDER_SEQ_COLUMNS))
    df["LogID"] = (df["LogID"].astype(str).str.strip()
                   .str.replace(r"\.0+$", "", regex=True))
    df["OrderTime"]      = pd.to_datetime(df["OrderTime"], errors="coerce")
    df["SeqInEncounter"] = pd.to_numeric(df["SeqInEncounter"], errors="coerce")
    return df


def collapse_order_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse consecutive identical OrderGroup per LogID into runs.

    Removes GNN self-loops and shortens sequences (avg ~303 raw), while
    preserving intensity as RepeatCount and the run's time span. Returns
    one row per run with a fresh SeqInEncounter + inter-run gap."""
    df = df.sort_values(["LogID", "SeqInEncounter"]).copy()
    df["_run"] = (df["OrderGroup"]
                  .ne(df.groupby("LogID")["OrderGroup"].shift())
                  .groupby(df["LogID"]).cumsum())
    runs = (df.groupby(["LogID", "_run"], sort=False)
              .agg(OrderGroup =("OrderGroup",  "first"),
                   OrderSource=("OrderSource", "first"),
                   RepeatCount=("OrderGroup",  "size"),
                   RunStart   =("OrderTime",   "first"),
                   RunEnd     =("OrderTime",   "last"))
              .reset_index()
              .drop(columns="_run"))
    runs["SeqInEncounter"]  = runs.groupby("LogID").cumcount() + 1
    runs["MinutesFromPrev"] = (runs.groupby("LogID")["RunStart"]
                               .diff().dt.total_seconds() / 60.0)
    return runs


# ---------------------------------------------------------------------
# Care-team provider block (Option B) -- encounter-level, from A2 + A4.
# No per-order attribution (ordering-provider is unreliable: orders are
# often placed under/cosigned by the attending, not the deciding resident).
# ---------------------------------------------------------------------
def _yn_to_int(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.strip().str.upper()
            .isin(["Y", "1", "TRUE", "YES"]).astype(int))


def build_provider_team_features(
    prov_edges: pd.DataFrame,     # A2: LogID, ProvID, ...
    prov_attrs: pd.DataFrame,     # A4: ProvID, IsResident, EmployedCRNA, ...
    encounters: pd.DataFrame,     # A1: LogID, PrimarySurgID, ...
) -> pd.DataFrame:
    """One row per LogID: care-team composition + primary-surgeon volume
    (Option B). Distinct from build_provider_features() below, which builds
    per-provider node features/embeddings for the GNN.
    Returns C.PROVIDER_FEATURE_COLUMNS (+ LogID). Missing -> 0."""
    _n = lambda s: s.astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)

    pe = prov_edges[["LogID", "ProvID"]].copy()
    pe["LogID"], pe["ProvID"] = _n(pe["LogID"]), _n(pe["ProvID"])

    pa = prov_attrs.copy()
    pa["ProvID"] = _n(pa["ProvID"])
    for c in ("IsResident", "EmployedCRNA", "IsHospitalist"):
        pa[c] = _yn_to_int(pa[c])
    for c in ("CaseVolume2yr", "CaseVolume5yr"):
        pa[c] = pd.to_numeric(pa[c], errors="coerce")

    # team composition
    m = pe.drop_duplicates(["LogID", "ProvID"]).merge(pa, on="ProvID", how="left")
    team = (m.groupby("LogID")
              .agg(team_size     =("ProvID",       "nunique"),
                   n_residents   =("IsResident",   "sum"),
                   n_crna        =("EmployedCRNA", "sum"),
                   n_hospitalists=("IsHospitalist","sum"))
              .reset_index())
    team["has_resident"]    = (team["n_residents"]    > 0).astype(int)
    team["has_crna"]        = (team["n_crna"]         > 0).astype(int)
    team["has_hospitalist"] = (team["n_hospitalists"] > 0).astype(int)

    # primary-surgeon volume (A1.PrimarySurgID -> A4.CaseVolume*)
    enc = encounters[["LogID", "PrimarySurgID"]].copy()
    enc["LogID"], enc["PrimarySurgID"] = _n(enc["LogID"]), _n(enc["PrimarySurgID"])
    surg = enc.merge(pa[["ProvID", "CaseVolume2yr", "CaseVolume5yr"]],
                     left_on="PrimarySurgID", right_on="ProvID", how="left")
    # unmatched = surgeon not found in A4 (no volume data) -- keep separate
    # from a matched low-volume surgeon (whose volume is legitimately small).
    surg["surg_vol_unmatched"] = surg["ProvID"].isna().astype(int)
    surg = surg.rename(columns={"CaseVolume2yr": "surg_casevol_2yr",
                                "CaseVolume5yr": "surg_casevol_5yr"})

    out = (encounters[["LogID"]].assign(LogID=lambda d: _n(d["LogID"]))
           .merge(team, on="LogID", how="left")
           .merge(surg[["LogID", "surg_casevol_2yr", "surg_casevol_5yr",
                        "surg_vol_unmatched"]], on="LogID", how="left"))
    out[list(C.PROVIDER_FEATURE_COLUMNS)] = (
        out[list(C.PROVIDER_FEATURE_COLUMNS)].apply(pd.to_numeric, errors="coerce").fillna(0)
    )
    return out


# ---------------------------------------------------------------------
# Encounter feature preprocessing
# ---------------------------------------------------------------------
@dataclass
class PreprocessState:
    """Everything needed to re-apply the train-fit preprocessing to
    val/test/inference. Pickled alongside the trained model."""
    continuous_cols: List[str]      = field(default_factory=list)
    categorical_cols: List[str]     = field(default_factory=list)
    continuous_means: Dict[str, float] = field(default_factory=dict)
    iqr_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    ohe_categories: Dict[str, List[str]] = field(default_factory=dict)
    scaler: Optional[StandardScaler] = None
    final_feature_names: List[str]  = field(default_factory=list)


def _identify_column_types(
    df: pd.DataFrame, id_cols: List[str]
) -> Tuple[List[str], List[str]]:
    """Split into continuous vs categorical.

    Numeric dtype + low-cardinality integer-valued -> categorical
    (e.g. ASAClass 1-5, comorbidity 0/1 flags). Everything else
    numeric -> continuous. Non-numeric -> categorical.
    """
    LOW_CARD_INT_LIMIT = 6
    continuous, categorical = [], []
    for col in df.columns:
        if col in id_cols:
            continue
        s = df[col].dropna()
        if not pd.api.types.is_numeric_dtype(df[col]):
            categorical.append(col)
            continue
        is_integer_valued = len(s) > 0 and bool(np.all(s.astype(int) == s))
        if is_integer_valued and s.nunique() <= LOW_CARD_INT_LIMIT:
            categorical.append(col)
        else:
            continuous.append(col)
    return continuous, categorical


def _iqr_clip(s: pd.Series, k: float = 1.5) -> Tuple[pd.Series, Tuple[float, float]]:
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return s.clip(lo, hi), (float(lo), float(hi))


def _apply_random_mask(
    df: pd.DataFrame, rates: Dict[str, float], seed: int = 0
) -> pd.DataFrame:
    """Reproduces the paper's missing-data ablation experiment."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    for col, rate in rates.items():
        if col not in out.columns or rate <= 0:
            continue
        mask = rng.random(len(out)) < rate
        out.loc[mask, col] = np.nan
    return out


def fit_preprocess(
    train_df: pd.DataFrame,
    id_cols: List[str],
    cfg: C.FeaturePrepConfig = C.DEFAULTS_FEATS,
) -> Tuple[np.ndarray, PreprocessState]:
    """Fit preprocessing on the train split. Returns the train feature
    matrix and the state needed to apply the same transform elsewhere."""
    state = PreprocessState()
    df = train_df.copy()

    state.continuous_cols, state.categorical_cols = _identify_column_types(df, id_cols)

    # 1. IQR clip on continuous - paper "filter outliers".
    if cfg.outlier_method == "iqr_clip":
        for col in state.continuous_cols:
            df[col], state.iqr_bounds[col] = _iqr_clip(
                pd.to_numeric(df[col], errors="coerce"), cfg.iqr_multiplier
            )

    # 2. Optional random masking ablation.
    if cfg.enable_random_masking:
        df = _apply_random_mask(df, cfg.masking_rates, seed=0)

    # 3. Imputation - "Unknown" categorical, mean continuous. Cast
    # categoricals to str FIRST so that low-cardinality numerics (e.g.
    # ASA 1-5) and a string "Unknown" sentinel coexist in one column
    # without breaking sorted() / Categorical comparison in step 4.
    for col in state.categorical_cols:
        df[col] = df[col].astype("object").where(df[col].notna(), other=np.nan)
        df[col] = df[col].astype("string").fillna(cfg.categorical_fill)
    for col in state.continuous_cols:
        mu = pd.to_numeric(df[col], errors="coerce").mean()
        state.continuous_means[col] = float(mu) if pd.notna(mu) else 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(state.continuous_means[col])

    # 4. One-hot encode. Cast to Categorical so the get_dummies output
    # columns include EVERY category seen at train time, not just the
    # ones present in this particular dataframe.
    for col in state.categorical_cols:
        state.ohe_categories[col] = sorted(df[col].astype(str).unique().tolist())
        df[col] = pd.Categorical(df[col].astype(str),
                                 categories=state.ohe_categories[col])
    encoded = pd.get_dummies(
        df[state.categorical_cols],
        columns=state.categorical_cols,
        drop_first=cfg.drop_first_in_ohe,
        dummy_na=False,
    )

    # 5. Standardize continuous (StandardScaler on train only).
    if cfg.standardize and state.continuous_cols:
        state.scaler = StandardScaler()
        cont = state.scaler.fit_transform(df[state.continuous_cols].values.astype(float))
    else:
        cont = df[state.continuous_cols].values.astype(float)

    cont_df = pd.DataFrame(cont, columns=state.continuous_cols, index=df.index)
    final   = pd.concat([cont_df, encoded], axis=1)
    state.final_feature_names = final.columns.tolist()
    return final.values.astype(np.float32), state


def apply_preprocess(
    df: pd.DataFrame,
    state: PreprocessState,
    cfg: C.FeaturePrepConfig = C.DEFAULTS_FEATS,
) -> np.ndarray:
    """Apply a train-fit PreprocessState to val/test/inference rows."""
    df = df.copy()

    # 1. IQR clip with stored bounds.
    for col, (lo, hi) in state.iqr_bounds.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(lo, hi)

    # 2. (Random masking is train-only.)

    # 3. Impute with stored means / "Unknown". Cast categoricals to str
    # so they match the str-cast vocabulary stored at fit time.
    for col in state.categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna(cfg.categorical_fill)
        else:
            df[col] = cfg.categorical_fill
    for col in state.continuous_cols:
        mu = state.continuous_means.get(col, 0.0)
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(mu)
        else:
            df[col] = mu

    # 4. One-hot with categories pinned to train-time vocabulary so the
    # column set is identical regardless of which categories happen to
    # appear in this split. Keep the Categorical dtype - DO NOT cast
    # back to object, since that drops the pinned categories and
    # get_dummies would then only emit columns for values actually
    # present in this batch.
    for col in state.categorical_cols:
        df[col] = pd.Categorical(df[col].astype(str),
                                 categories=state.ohe_categories[col])
    encoded = pd.get_dummies(
        df[state.categorical_cols],
        columns=state.categorical_cols,
        drop_first=cfg.drop_first_in_ohe,
        dummy_na=False,
    )

    # 5. Standardize.
    if state.scaler is not None and state.continuous_cols:
        cont = state.scaler.transform(df[state.continuous_cols].values.astype(float))
    else:
        cont = df[state.continuous_cols].values.astype(float)

    cont_df = pd.DataFrame(cont, columns=state.continuous_cols, index=df.index)
    final = pd.concat([cont_df, encoded], axis=1)
    # Align to training feature order; any new categories become extra
    # columns that we drop; missing ones become zeros.
    final = final.reindex(columns=state.final_feature_names, fill_value=0)
    return final.values.astype(np.float32)


# ---------------------------------------------------------------------
# Node-feature builders for the three node types
# ---------------------------------------------------------------------
def build_encounter_features(
    raw: RawExtract,
    state_train: Optional[PreprocessState] = None,
    cfg: C.FeaturePrepConfig = C.DEFAULTS_FEATS,
) -> Tuple[pd.DataFrame, np.ndarray, Optional[PreprocessState]]:
    """Join A1 (encounter ids) with the NSQIP tabular features and the
    readmission label, then preprocess.

    If `state_train` is None, fits on this DataFrame (use only for the
    training split). Otherwise applies the stored transform.
    """
    merged = (raw.encounters
              .merge(raw.enc_features, on="LogID", how="inner")
              .merge(raw.labels[["LogID", "ReadmittedWithin30Days"]],
                     on="LogID", how="inner"))

    id_cols = ["LogID", "EncounterCSN", "PAT_ID", "SurgeryDate",
               "PrimarySurgID", "ReadmittedWithin30Days"]
    feat_df = merged.drop(columns=[c for c in id_cols
                                   if c in merged.columns
                                   and c != "ReadmittedWithin30Days"],
                          errors="ignore")
    feat_df = feat_df.drop(columns=["ReadmittedWithin30Days"], errors="ignore")

    if state_train is None:
        X, state = fit_preprocess(feat_df, id_cols=[], cfg=cfg)
        return merged, X, state
    X = apply_preprocess(feat_df, state_train, cfg=cfg)
    return merged, X, None


# ---------------------------------------------------------------------
# Encounter feature augmentation from the graph extracts (paper H^ENC
# "transfer history" + calendar blocks). Kept separate from the NSQIP
# tabular block so the leakage allow-list stays auditable.
# ---------------------------------------------------------------------
_PREOP_WARD_TYPES = ("Acute", "Intensive", "Intermediate")


def build_preop_trajectory_features(
    enc_unit_edges: pd.DataFrame,
    surgery_start: pd.DataFrame,
) -> pd.DataFrame:
    """Per-encounter PRE-OPERATIVE care-unit trajectory features from A3.

    A unit visit counts as pre-operative when it starts before the index
    surgery start time; for a visit that straddles surgery start we take
    only the hours up to surgery start. This keeps the feature strictly
    leakage-safe (no post-discharge / post-op information).

    Args:
        enc_unit_edges: A3 (LogID, UnitType, InTime, OutTime, DepartmentID...).
        surgery_start:  DataFrame with columns ['LogID', '_ss'] where _ss
                        is the parsed surgery-start datetime.

    Returns one row per LogID with the columns in
    config.TRAJECTORY_FEATURE_COLUMNS (zeros where no pre-op ward stay).
    """
    a3 = enc_unit_edges.copy()
    a3["LogID"] = a3["LogID"].astype(str)
    a3["_in"]  = pd.to_datetime(a3["InTime"],  errors="coerce")
    a3["_out"] = pd.to_datetime(a3["OutTime"], errors="coerce")

    ss = surgery_start.copy()
    ss["LogID"] = ss["LogID"].astype(str)
    a3 = a3.merge(ss[["LogID", "_ss"]], on="LogID", how="inner")

    # Pre-op overlap hours = [_in, min(_out, _ss)] clipped at >= 0,
    # only for visits that began before surgery start.
    preop_end = a3[["_out", "_ss"]].min(axis=1)
    a3["_preop_hr"] = ((preop_end - a3["_in"]).dt.total_seconds() / 3600.0)
    a3["_preop_hr"] = a3["_preop_hr"].clip(lower=0.0)
    a3.loc[a3["_in"] >= a3["_ss"], "_preop_hr"] = 0.0

    ward = a3[(a3["_in"] < a3["_ss"])
              & (a3["UnitType"].isin(_PREOP_WARD_TYPES))].copy()

    base = pd.DataFrame({"LogID": ss["LogID"].drop_duplicates().values})

    # LOS hours per ward type.
    los = (ward.groupby(["LogID", "UnitType"])["_preop_hr"].sum()
                .unstack(fill_value=0.0))
    for ut in _PREOP_WARD_TYPES:
        col = f"preop_los_{ut.lower()}_hr"
        base[col] = (base["LogID"].map(los[ut]) if ut in los.columns
                     else 0.0)
        base[col] = base[col].fillna(0.0).astype(float)

    # Pre-op ward-visit count (transfer frequency proxy) + distinct depts.
    visit_ct = ward.groupby("LogID").size()
    base["preop_transfer_count"] = (base["LogID"].map(visit_ct)
                                    .fillna(0).astype(int))
    dept_col = "DepartmentID" if "DepartmentID" in ward.columns else "UnitType"
    n_units = ward.groupby("LogID")[dept_col].nunique()
    base["preop_n_units"] = base["LogID"].map(n_units).fillna(0).astype(int)
    return base


def add_calendar_features(
    df: pd.DataFrame,
    start_col: str = "Procedure/Surgery Start",
    date_col:  str = "SurgeryDate",
) -> pd.DataFrame:
    """Add the paper's calendar block (day-of-week + weekend flag) from
    the surgery start timestamp, falling back to SurgeryDate. Returns a
    copy with `surgery_dow` (string, one-hot downstream) and
    `surgery_is_weekend` (0/1) added."""
    out = df.copy()
    src = start_col if start_col in out.columns else date_col
    dt = pd.to_datetime(out[src], errors="coerce")
    out["surgery_dow"] = dt.dt.day_name().astype("object")
    out["surgery_is_weekend"] = (dt.dt.dayofweek >= 5).astype("Int64").astype(float)
    return out


def build_provider_features(
    prov_attrs: pd.DataFrame,
    cfg: C.FeaturePrepConfig = C.DEFAULTS_FEATS,
) -> Tuple[pd.DataFrame, np.ndarray, PreprocessState]:
    """Numeric features for provider nodes. Keep low-cardinality
    categoricals (ProviderType, Specialty, flag columns); exclude
    high-cardinality / unhandled-dtype columns that would blow up the
    one-hot dimensionality and dilute the GNN signal.

    Excluded:
      - ID-shaped columns (ProvID, ProvName, NPI, PrimaryDeptID,
        PrimaryDeptName) -- not features, would one-hot to thousands of
        nearly-empty columns
      - ProvType -- raw 100+ category text; we already bucket it into
        ProviderType (5 cats) which is what we want
      - ProvStartDate -- raw datetime, would one-hot per timestamp.
        Convert below to YearsExperience as a continuous feature.
    """
    exclude = [
        "ProvID", "ProvName", "NPI",
        "PrimaryDeptID", "PrimaryDeptName",
        "ProvType",
        "ProvStartDate",
    ]
    feat_df = prov_attrs.drop(
        columns=[c for c in exclude if c in prov_attrs.columns],
        errors="ignore",
    )

    # Convert PROV_START_DATE to years-of-experience (continuous).
    if "ProvStartDate" in prov_attrs.columns:
        sd = pd.to_datetime(prov_attrs["ProvStartDate"], errors="coerce")
        # Anchor to the cohort's median surgery year (2023) to avoid
        # leaking 'today' into the training fold. The exact reference
        # doesn't matter for relative-experience ranking.
        ref = pd.Timestamp("2023-01-01")
        feat_df = feat_df.assign(
            YearsExperience=((ref - sd).dt.days / 365.25)
                              .clip(lower=0, upper=60)
                              .fillna(0)
        )

    X, state = fit_preprocess(feat_df, id_cols=[], cfg=cfg)
    return prov_attrs[["ProvID"]].reset_index(drop=True), X, state


def build_unit_features(
    unit_attrs: pd.DataFrame,
    cfg: C.FeaturePrepConfig = C.DEFAULTS_FEATS,
) -> Tuple[pd.DataFrame, np.ndarray, PreprocessState]:
    """Unit features - one-hot of UnitType + DepartmentName.

    NOTE: we tried excluding DepartmentName (231 dims) and ServiceAreaID
    on the theory that high-cardinality columns hurt the GNN. Both
    experiments lost AUROC (~0.008-0.016 per col removed on single runs).
    The unit table has only 231 rows, so every column effectively
    contributes per-unit learnable features. Keep them all -- the
    graph_id JOIN already strips out DepartmentID/LocationID which are
    the only true non-features.
    """
    id_cols = ["DepartmentID", "LocationID"]
    feat_df = unit_attrs.drop(columns=[c for c in id_cols
                                       if c in unit_attrs.columns],
                              errors="ignore")
    X, state = fit_preprocess(feat_df, id_cols=[], cfg=cfg)
    return unit_attrs[["DepartmentID"]].reset_index(drop=True), X, state


# ---------------------------------------------------------------------
# Train / val / test split (paper: 8:1:1)
# ---------------------------------------------------------------------
def stratified_split(
    encounters_merged: pd.DataFrame,
    cfg: C.TrainConfig = C.DEFAULTS_TRAIN,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified 8:1:1 split by the binary readmission label.
    Returns boolean masks indexed against `encounters_merged`."""
    rng = np.random.default_rng(cfg.split_seed)
    y = encounters_merged["ReadmittedWithin30Days"].astype(int).values
    n = len(y)
    idx = np.arange(n)
    train, val, test = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
    for cls in (0, 1):
        cls_idx = idx[y == cls]
        rng.shuffle(cls_idx)
        n_train = int(round(cfg.train_frac * len(cls_idx)))
        n_val   = int(round(cfg.val_frac   * len(cls_idx)))
        train[cls_idx[:n_train]]                    = True
        val  [cls_idx[n_train:n_train + n_val]]     = True
        test [cls_idx[n_train + n_val:]]            = True
    return train, val, test


# ---------------------------------------------------------------------
# Persist / restore the entire preprocessing state
# ---------------------------------------------------------------------
def save_preprocess(state: PreprocessState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(state, f)


def load_preprocess(path: Path) -> PreprocessState:
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------
# ZIP -> nearest-hospital distance (LLM-explanation driver).
# NOT a scoring feature -- population lift is negligible (~+0.002 AUROC).
# Used to surface an access-of-care driver in the CDS explanation sentence,
# per clinical review. Distance is haversine miles between the ZIP centroid
# (pgeocode US ZCTA) and the nearest anchor hospital.
# ---------------------------------------------------------------------

# WCM/NYP anchor hospitals (lat, lon). Add more sites here as needed.
HOSPITAL_ANCHORS = {
    "GBG_WCM_main": (40.7651, -73.9638),      # Greenberg / WCM main, 10065
    "LMH":          (40.7095, -74.0135),      # Lower Manhattan, 10038
}

# Clinical-review driver thresholds.
#   >5 mi     : the reviewer-proposed cutoff (access-of-care risk band starts here)
#   >25 mi    : capture-bias caveat -- readmit rate drops (5.7%) because distant
#               patients readmit to non-WCM facilities the EHR doesn't see;
#               so the actionable risk band is 5-25 mi (~9.8% readmit vs 7.7%
#               near-anchor).
FAR_THRESHOLD_MI = 5.0
CAPTURE_BIAS_MI = 25.0


def _haversine_miles(lat1, lon1, lat2, lon2) -> np.ndarray:
    R = 3958.8
    p = np.pi / 180.0
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


def build_zip_distance_features(
    df: pd.DataFrame,
    zip_col: str = "ZIP",
    anchors: Optional[Dict[str, Tuple[float, float]]] = None,
) -> pd.DataFrame:
    """LLM-explanation driver: haversine distance from each patient's ZIP to
    the nearest NYP anchor hospital, plus reviewer-proposed access-band flags.

    Returns a DataFrame with, per row:
      * zip_dist_mi         distance to nearest anchor (NaN if ZIP missing/invalid)
      * zip_far_gt5mi       1 if > 5 mi from nearest anchor
      * zip_risk_band_5_25  1 if 5-25 mi (the actionable band; > 25 mi drops
                            due to out-of-network readmission capture bias, so
                            this refines the raw >5 mi cutoff)

    Not a scoring feature -- adds ~+0.002 AUROC to RF tabular -- but retained for
    the CDS explanation layer per clinical review (Clinical_Review_Driver_Thresholds).
    Requires `pgeocode`.
    """
    import pgeocode                                        # lazy import (optional dep)
    a = anchors or HOSPITAL_ANCHORS
    out = df[[zip_col]].copy() if zip_col in df.columns else pd.DataFrame({zip_col: [np.nan] * len(df)})
    out["zip5"] = out[zip_col].astype(str).str.extract(r"^(\d{5})", expand=False)
    uniq = out["zip5"].dropna().unique().tolist()
    if uniq:
        geo = pgeocode.Nominatim("us").query_postal_code(uniq)[
            ["postal_code", "latitude", "longitude"]
        ].rename(columns={"postal_code": "zip5"})
        geo["zip5"] = geo["zip5"].astype(str)
        out = out.merge(geo, on="zip5", how="left")
    else:
        out["latitude"] = np.nan
        out["longitude"] = np.nan
    dists = np.column_stack([
        _haversine_miles(out["latitude"].values, out["longitude"].values, lat, lon)
        for (lat, lon) in a.values()
    ])
    out["zip_dist_mi"] = np.nanmin(dists, axis=1) if dists.size else np.nan
    out["zip_far_gt5mi"] = (out["zip_dist_mi"] > FAR_THRESHOLD_MI).astype("Int64")
    out["zip_risk_band_5_25"] = (
        (out["zip_dist_mi"] > FAR_THRESHOLD_MI)
        & (out["zip_dist_mi"] <= CAPTURE_BIAS_MI)
    ).astype("Int64")
    mask = out["zip_dist_mi"].isna()
    for col in ("zip_far_gt5mi", "zip_risk_band_5_25"):
        out.loc[mask, col] = pd.NA
    return out[["zip_dist_mi", "zip_far_gt5mi", "zip_risk_band_5_25"]]
