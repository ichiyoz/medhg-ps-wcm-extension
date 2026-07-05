"""Deployment model: the best CV-validated readmission predictor.

The relational/GNN exploration was negative (see analysis/cv_*.py and the
manuscript): no graph model beat a tree. The deployable model is therefore a
plain gradient-boosted / random-forest tree on the enriched tabular feature set:

    NSQIP tabular features  (+ pre-op trajectory + calendar derived)
    + PrimaryCPT            (one-hot)
    + post-op care-unit sequence features

5-fold CV performance (base rate 9.58%):  AUROC ~0.700,  AUPRC ~0.235-0.238.

This module is torch-free and depends only on scikit-learn / pandas / numpy so
the serialized bundle loads in a minimal production environment. The fitted
`ReadmissionModel` is what gets pickled; it carries every fitted transform plus
a `.predict()` that takes raw per-encounter records.

Build/export with:   PYTHONPATH=. python analysis/export_model.py
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import medhg_ps.config as C
from medhg_ps.data import (PreprocessState, add_calendar_features, apply_preprocess,
                           build_preop_trajectory_features, fit_preprocess, load_raw)

# --- post-op care-unit sequence featurisation (shared train/inference) -------
SEQ_UNITS = ("ED", "Acute", "OR", "Intensive", "Intermediate", "Other")
# Clarity UnitType -> coarse GNN bucket (same mapping used across analysis/)
UNIT_BUCKET = {"ICU": "Intensive", "PICU": "Intensive", "NICU": "Intensive",
               "Med/Surg": "Acute", "Procedural Area": "OR", "ED": "ED",
               "Recovery Area": "Intermediate"}
MIN_SUP = 50            # keep a sequence feature only if >=50 train encounters have it
COHORT_COLS = ["LogID", "EncounterCSN", "PAT_ID", "SurgeryDate", "SurgeryYear",
               "PrimaryCPT", "AgeYears"]


def _collapse(seq: Sequence[str]) -> List[str]:
    """Collapse consecutive duplicates: [A,A,B,B,A] -> [A,B,A]."""
    out: List[str] = []
    for x in seq:
        if not out or out[-1] != x:
            out.append(x)
    return out


def seq_feature_dict(unit_seq: Sequence[str]) -> Dict[str, float]:
    """All candidate post-op sequence features for one encounter's unit path.

    `unit_seq` is the time-ordered list of coarse care-unit buckets the patient
    occupied during the index admission (raw, with repeats). Already-bucketed
    values (ED/Acute/OR/Intensive/Intermediate/Other) are expected; unknown
    strings fall through as their own (rare) tokens and are dropped by the
    support filter at fit time.
    """
    rawseq = list(unit_seq)
    f: Dict[str, float] = {}
    for un in SEQ_UNITS:
        f[f"cnt_{un}"] = float(rawseq.count(un))
    c = _collapse(rawseq)
    if not c:
        return f
    f["n_stops"] = float(len(c))
    f["n_or"] = float(c.count("OR"))
    f["has_ICU"] = float("Intensive" in c)
    f["ICU_after_OR"] = float("Intensive" in c and "OR" in c
                              and c.index("Intensive") > c.index("OR"))
    f[f"end_{c[-1]}"] = 1.0
    for a, b in zip(c, c[1:]):
        f[f"bg_{a}>{b}"] = f.get(f"bg_{a}>{b}", 0.0) + 1.0
    return f


def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)


@dataclass
class ReadmissionModel:
    """Self-contained, picklable readmission predictor.

    Holds every fitted transform plus the classifier. Call `.predict(records)`
    with raw per-encounter dicts to get readmission probabilities.
    """
    estimator_name: str                    # "hgb" or "rf"
    clf: Any                               # fitted (optionally calibrated) classifier
    preprocess_state: PreprocessState      # tabular NSQIP transform
    tab_feat_cols: List[str]               # tabular columns fed to apply_preprocess
    cpt_encoder: OneHotEncoder             # PrimaryCPT one-hot
    seq_keep: List[str]                    # post-op sequence columns kept (support-filtered)
    scaler: StandardScaler                 # over [tab | cpt | seq]
    n_features: int                        # width of the final vector
    cv_metrics: Dict[str, float] = field(default_factory=dict)
    base_rate: float = 0.0
    threshold: float = 0.5                 # default operating point (override per deployment)
    version: str = "1.0"

    # --- feature assembly for inference -----------------------------------
    def _vectorize(self, records: List[Dict[str, Any]]) -> np.ndarray:
        df = pd.DataFrame(records)
        # tabular block (apply_preprocess fills any missing columns)
        tab_in = df.reindex(columns=self.tab_feat_cols)
        Xtab = apply_preprocess(tab_in, self.preprocess_state)
        # CPT one-hot
        if "PrimaryCPT" in df.columns:
            cpt = _norm(df["PrimaryCPT"]).fillna("UNK")
        else:
            cpt = pd.Series(["UNK"] * len(df))
        cpt_arr = np.asarray(cpt.astype(str), dtype=object).reshape(-1, 1)
        Xcpt = self.cpt_encoder.transform(cpt_arr)
        # post-op sequence block
        seq_rows = []
        seqs = df["postop_units"] if "postop_units" in df.columns else [[]] * len(df)
        for s in seqs:
            d = seq_feature_dict(s if isinstance(s, (list, tuple)) else [])
            seq_rows.append([d.get(c, 0.0) for c in self.seq_keep])
        Xseq = np.asarray(seq_rows, dtype=float).reshape(len(df), len(self.seq_keep))
        X = np.hstack([Xtab, Xcpt, Xseq])
        return self.scaler.transform(X)

    def predict_proba(self, records: List[Dict[str, Any]]) -> np.ndarray:
        """Readmission probability for each record (1-D array)."""
        if isinstance(records, dict):
            records = [records]
        return self.clf.predict_proba(self._vectorize(records))[:, 1]

    def predict(self, records: List[Dict[str, Any]],
                threshold: Optional[float] = None) -> np.ndarray:
        """Binary readmission flag at `threshold` (defaults to self.threshold)."""
        t = self.threshold if threshold is None else threshold
        return (self.predict_proba(records) >= t).astype(int)


# --- training / assembly -----------------------------------------------------
def _load_cpt_map() -> Dict[str, str]:
    for d in [str(C.DATA_DIR), "/Users/yiyezhang/Dropbox/Surgery"]:
        for p in glob.glob(os.path.join(d, "*.csv")):
            try:
                h = pd.read_csv(p, nrows=1, header=None, dtype=str)
            except Exception:
                continue
            if str(h.iloc[0, 0]).strip() == "LogID":
                df = pd.read_csv(p, low_memory=False)
                if {"LogID", "PrimaryCPT"} <= set(df.columns):
                    return dict(zip(_norm(df["LogID"]), _norm(df["PrimaryCPT"])))
            elif h.shape[1] == len(COHORT_COLS):
                df = pd.read_csv(p, header=None, names=COHORT_COLS, low_memory=False)
                return dict(zip(_norm(df["LogID"]), _norm(df["PrimaryCPT"])))
    raise FileNotFoundError("cohort CSV with PrimaryCPT not found")


def assemble_training_frame():
    """Reconstruct the enriched modelling frame (matches cv_allfeat_relational).

    Returns (merged_df, feat_cols, cpt_array, Fseq_df, seq_all_cols, y).
    """
    cpt_map = _load_cpt_map()
    raw = load_raw()
    enc_nodupes = raw.enc_features.drop(
        columns=([c for c in raw.encounters.columns
                  if c != "LogID" and c in raw.enc_features.columns]
                 + ["ReadmittedWithin30Days"]), errors="ignore")
    merged = (raw.encounters.merge(enc_nodupes, on="LogID", how="inner")
              .merge(raw.labels[["LogID", "ReadmittedWithin30Days"]], on="LogID", how="inner")
              .reset_index(drop=True))
    ss = merged[["LogID"]].copy()
    ss["_ss"] = pd.to_datetime(merged.get("Procedure/Surgery Start"), errors="coerce")
    merged = merged.merge(build_preop_trajectory_features(raw.enc_unit_edges, ss),
                          on="LogID", how="left")
    for c in C.TRAJECTORY_FEATURE_COLUMNS:
        merged[c] = merged[c].fillna(0)
    merged = add_calendar_features(merged)
    merged["LogID"] = merged["LogID"].astype(str)
    feat_cols = ([c for c in C.MODEL_FEATURE_COLUMNS if c in merged.columns]
                 + [c for c in C.ENCOUNTER_DERIVED_COLUMNS if c in merged.columns])
    y = merged["ReadmittedWithin30Days"].astype(int).values
    cpt_arr = np.array([cpt_map.get(l, "UNK") for l in merged["LogID"]]).reshape(-1, 1)

    # full post-op care-unit sequence features
    u = pd.read_excel(Path(__file__).resolve().parents[1] / "analysis" / "Unit_Names.xlsx")
    u["cid"] = u["Clarity_ID"].astype("Int64").astype(str)
    u["g"] = u["UnitType"].map(UNIT_BUCKET).fillna("Other")
    ud = u.dropna(subset=["Clarity_ID"]).drop_duplicates("cid", keep="first").set_index("cid")
    a3 = raw.enc_unit_edges.copy()
    a3["UnitType"] = (a3["DepartmentID"].astype(str).str.replace(r"\.0+$", "", regex=True)
                      .map(ud["g"]).fillna(a3["UnitType"]))
    a3["InTime"] = pd.to_datetime(a3["InTime"]); a3["LogID"] = a3["LogID"].astype(str)
    seqs = a3.sort_values(["LogID", "InTime"]).groupby("LogID", sort=False)["UnitType"].apply(list)
    rows = []
    for lid, rawseq in seqs.items():
        d = seq_feature_dict(rawseq); d["LogID"] = lid
        rows.append(d)
    Fseq = pd.DataFrame(rows).fillna(0)
    Fseq["LogID"] = Fseq["LogID"].astype(str)
    Fseq = merged[["LogID"]].merge(Fseq, on="LogID", how="left").fillna(0)
    seq_all = [c for c in Fseq.columns if c != "LogID"]
    return merged, feat_cols, cpt_arr, Fseq, seq_all, y


def _make_estimator(name: str):
    if name == "rf":
        return RandomForestClassifier(n_estimators=500, min_samples_leaf=10,
                                      max_features="sqrt", class_weight="balanced",
                                      random_state=42, n_jobs=-1)
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                          l2_regularization=1.0, random_state=42)


def fit_full(merged, feat_cols, cpt_arr, Fseq, seq_all, y, estimator_name: str):
    """Fit the entire pipeline on ALL rows; return a ReadmissionModel (uncalibrated clf)."""
    idx = np.arange(len(merged))
    Xtab, st = fit_preprocess(merged[feat_cols].reset_index(drop=True), id_cols=[])
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(cpt_arr)
    Xcpt = ohe.transform(cpt_arr)
    keep = [c for c in seq_all if (Fseq[c] > 0).sum() >= MIN_SUP]
    Xseq = Fseq[keep].values.astype(float)
    Xall = np.hstack([Xtab, Xcpt, Xseq])
    sc = StandardScaler().fit(Xall)
    Xall = sc.transform(Xall)
    clf = _make_estimator(estimator_name).fit(Xall, y)
    return ReadmissionModel(
        estimator_name=estimator_name, clf=clf, preprocess_state=st,
        tab_feat_cols=list(feat_cols), cpt_encoder=ohe, seq_keep=keep, scaler=sc,
        n_features=Xall.shape[1], base_rate=float(y.mean()),
    )
