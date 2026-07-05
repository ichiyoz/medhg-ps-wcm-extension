"""
deploy_emb.py — Lookup-only deployment wrapper for the MedHG-PS embedding model.

V2: now includes CPT and diagnosis embedding contributions in addition to
provider and unit. The HGT was trained over 4 neighbor entity types
(provider / unit / CPT / dx); pooling all four is the right approximation
of the trained encounter embedding for lookup-only inference.

Inference contract:
    [ tabular NSQIP | CPT one-hot | post-op care-unit sequence
      | emb_scaler.transform( mean( provider_emb, unit_emb, cpt_emb, dx_emb ) ) ]
        -> calibrated HistGradientBoosting -> 30-day readmission probability

Per-record input keys consumed by predict_proba:
    provider_ids : list[str]   surgeon SURG_IDs for this case
    unit_ids     : list[str]   care-unit DEPARTMENT_IDs (post-op trajectory)
    cpt_ids      : list[str]   primary (and optionally secondary) CPT codes
    dx_ids       : list[str]   principal (and optionally secondary) Dx codes
    PrimaryCPT   : str         (separate, for the existing cpt_encoder one-hot)
    postop_units : list[str]   (separate, for the existing sequence featurisation)
    + all NSQIP tabular fields the trained preprocess_state expects.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import medhg_ps.config as C
from medhg_ps.data import PreprocessState, apply_preprocess


# Lazy / optional import — only needed for transductive (new-node) path
try:
    import dgl  # noqa: F401
    _DGL_AVAILABLE = True
except ImportError:
    dgl = None
    _DGL_AVAILABLE = False


def _require_dgl() -> None:
    if not _DGL_AVAILABLE:
        raise ImportError(
            "dgl is required for transductive (new-node) graph propagation "
            "but is not installed. The lookup-only inference path does not need it."
        )


SEQ_UNITS = ("ED", "Acute", "OR", "Intensive", "Intermediate", "Other")


# ---------------------------------------------------------------------
# Helpers — sequence featurisation
# ---------------------------------------------------------------------
def _collapse(seq: Sequence[str]) -> List[str]:
    out: List[str] = []
    for x in seq:
        if not out or out[-1] != x:
            out.append(x)
    return out


def seq_feature_dict(unit_seq: Sequence[str]) -> Dict[str, float]:
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
    f["ICU_after_OR"] = float(
        "Intensive" in c and "OR" in c and c.index("Intensive") > c.index("OR")
    )
    f[f"end_{c[-1]}"] = 1.0
    for a, b in zip(c, c[1:]):
        f[f"bg_{a}>{b}"] = f.get(f"bg_{a}>{b}", 0.0) + 1.0
    return f


def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)


# ---------------------------------------------------------------------
# Embedding lookup
# ---------------------------------------------------------------------
def _lookup_embedding(
    ids: Sequence[str],
    emb_table: pd.DataFrame,
    type_avg_table: Optional[pd.DataFrame] = None,
    types: Optional[Sequence[str]] = None,
    id_col: str = "EntityID",
    dim_prefix: str = "emb_",
) -> np.ndarray:
    """Mean-pool the embeddings for a list of entity IDs (providers, units,
    CPTs, or diagnoses). Unknown IDs fall back to per-type-average if
    supplied; otherwise contribute a zero vector. If every ID is unknown
    and no type info given, returns all-zero."""
    emb_cols = [c for c in emb_table.columns if c.startswith(dim_prefix)]
    if not emb_cols:
        raise ValueError(f"Embedding table has no columns starting with {dim_prefix!r}")
    n_dim = len(emb_cols)

    if not ids:
        return np.zeros(n_dim, dtype=float)
    ids_str = [str(i).strip() for i in ids if str(i).strip()]
    if not ids_str:
        return np.zeros(n_dim, dtype=float)

    table_idx = emb_table.set_index(id_col) if id_col in emb_table.columns else emb_table
    hits = table_idx.reindex(ids_str)
    found_mask = hits[emb_cols[0]].notna() if emb_cols else pd.Series(False, index=hits.index)

    vecs: List[np.ndarray] = []
    for i, _id in enumerate(ids_str):
        if found_mask.iloc[i]:
            vecs.append(hits.iloc[i][emb_cols].to_numpy(dtype=float))
            continue
        if (type_avg_table is not None
                and not type_avg_table.empty
                and types is not None and i < len(types)):
            ttype = str(types[i]).strip()
            if ttype and ttype in type_avg_table.index:
                vecs.append(type_avg_table.loc[ttype, emb_cols].to_numpy(dtype=float))
                continue
        vecs.append(np.zeros(n_dim, dtype=float))

    arr = np.vstack(vecs) if vecs else np.zeros((1, n_dim), dtype=float)
    return arr.mean(axis=0)


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
@dataclass
class ReadmissionEmbModel:
    """Lookup-only readmission predictor with bundled embedding tables for
    all four entity types (provider / unit / CPT / dx). Pure NumPy/pandas/
    sklearn at inference."""

    estimator_name: str
    arch: str
    clf: Any
    preprocess_state: PreprocessState
    tab_feat_cols: List[str]
    cpt_encoder: OneHotEncoder
    seq_keep: List[str]
    scaler: StandardScaler

    # Bundled embedding tables — one per entity type
    provider_embeddings: pd.DataFrame
    unit_embeddings:     pd.DataFrame
    cpt_embeddings:      pd.DataFrame = field(default_factory=pd.DataFrame)
    dx_embeddings:       pd.DataFrame = field(default_factory=pd.DataFrame)
    provider_type_avg:   pd.DataFrame = field(default_factory=pd.DataFrame)
    unit_type_avg:       pd.DataFrame = field(default_factory=pd.DataFrame)
    cpt_type_avg:        pd.DataFrame = field(default_factory=pd.DataFrame)
    dx_type_avg:         pd.DataFrame = field(default_factory=pd.DataFrame)

    n_features: int = 0
    n_total: int = 0
    cv_metrics: Dict[str, float] = field(default_factory=dict)
    base_rate: float = 0.0
    threshold: float = 0.5
    version: str = "1.0"

    # ----- inference -----
    def _vectorize(self, records: List[Dict[str, Any]]) -> np.ndarray:
        df = pd.DataFrame(records)

        # tabular block
        tab_in = df.reindex(columns=self.tab_feat_cols)
        Xtab = apply_preprocess(tab_in, self.preprocess_state)

        # CPT one-hot (separate from CPT graph embedding lookup)
        if "PrimaryCPT" in df.columns:
            cpt = _norm(df["PrimaryCPT"]).fillna("UNK")
        else:
            cpt = pd.Series(["UNK"] * len(df))
        Xcpt = self.cpt_encoder.transform(
            np.asarray(cpt.astype(str), dtype=object).reshape(-1, 1)
        )

        # post-op sequence block
        seq_rows = []
        seqs = df["postop_units"] if "postop_units" in df.columns else [[]] * len(df)
        for s in seqs:
            d = seq_feature_dict(s if isinstance(s, (list, tuple)) else [])
            seq_rows.append([d.get(c, 0.0) for c in self.seq_keep])
        Xseq = np.asarray(seq_rows, dtype=float).reshape(len(df), len(self.seq_keep))

        # 4-entity embedding lookup + mean-pool, approximating the trained
        # HGT's neighbor aggregation. emb_scaler was fit on per-encounter
        # 64-d embeddings at training, so it takes the pooled 64-d vector.
        emb_dim = sum(1 for c in self.provider_embeddings.columns if c.startswith("emb_"))
        N = len(df)
        prov_rows = np.zeros((N, emb_dim), dtype=float)
        unit_rows = np.zeros((N, emb_dim), dtype=float)
        cpt_rows  = np.zeros((N, emb_dim), dtype=float)
        dx_rows   = np.zeros((N, emb_dim), dtype=float)

        for i, rec in enumerate(records):
            prov_ids = rec.get("provider_ids") or []
            unit_ids = rec.get("unit_ids")     or []
            cpt_ids  = rec.get("cpt_ids")      or (
                [rec["PrimaryCPT"]] if rec.get("PrimaryCPT") else []
            )
            dx_ids   = rec.get("dx_ids")       or []

            prov_rows[i] = _lookup_embedding(
                prov_ids, self.provider_embeddings,
                type_avg_table=(self.provider_type_avg
                                if not self.provider_type_avg.empty else None),
            )
            unit_rows[i] = _lookup_embedding(
                unit_ids, self.unit_embeddings,
                type_avg_table=(self.unit_type_avg
                                if not self.unit_type_avg.empty else None),
            )
            cpt_rows[i] = _lookup_embedding(
                cpt_ids, self.cpt_embeddings,
                type_avg_table=(self.cpt_type_avg
                                if not self.cpt_type_avg.empty else None),
            ) if not self.cpt_embeddings.empty else np.zeros(emb_dim)
            dx_rows[i] = _lookup_embedding(
                dx_ids, self.dx_embeddings,
                type_avg_table=(self.dx_type_avg
                                if not self.dx_type_avg.empty else None),
            ) if not self.dx_embeddings.empty else np.zeros(emb_dim)

        # Mean across all 4 entity types — equal weighting approximation
        # of the HGT's attention-weighted aggregation
        encounter_emb_approx = (prov_rows + unit_rows + cpt_rows + dx_rows) / 4.0
        scaled_emb = self.scaler.transform(encounter_emb_approx)

        return np.hstack([Xtab, Xcpt, Xseq, scaled_emb])

    def predict_proba(self, records: List[Dict[str, Any]]) -> np.ndarray:
        if isinstance(records, dict):
            records = [records]
        return self.clf.predict_proba(self._vectorize(records))[:, 1]

    def predict(
        self,
        records: List[Dict[str, Any]],
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        t = self.threshold if threshold is None else threshold
        return (self.predict_proba(records) >= t).astype(int)
