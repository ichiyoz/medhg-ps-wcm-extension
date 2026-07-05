"""
Embedding extraction for downstream Random Forest integration.

After training MedHG-PS, we want to ship two parquet files with the
final-layer (H^X_3) embeddings for every provider and every care unit
in the training graph. The deployment script (PeriOp_NSQIP_model.py)
loads these at predict time, looks up the providers and unit
trajectory returned by the runtime SQL (Part B of GNN_Graph_Extract.sql),
and averages the embeddings before concatenating with the tabular
feature vector for the Random Forest.

Cold-start: any ProvID / DepartmentID not present in the embedding
table is replaced with the type-average embedding, also written here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from . import config as C
from .graph import GraphArtifacts
from .model import MedHGPS


# =====================================================================
@dataclass
class EmbeddingTables:
    provider:        pd.DataFrame   # ProvID + emb_0..emb_{d-1}
    unit:            pd.DataFrame   # DepartmentID + emb_0..emb_{d-1}
    encounter:       pd.DataFrame   # LogID + emb_0..emb_{d-1}
    provider_type_avg: pd.DataFrame # ProviderType -> mean emb (cold-start)
    unit_type_avg:     pd.DataFrame # UnitType     -> mean emb (cold-start)


# =====================================================================
def extract_embeddings(
    model: MedHGPS,
    artifacts: GraphArtifacts,
    raw_prov_attrs: Optional[pd.DataFrame] = None,
    raw_unit_attrs: Optional[pd.DataFrame] = None,
    device: str = "cpu",
) -> EmbeddingTables:
    model = model.to(device).eval()
    g = artifacts.g.to(device)
    fd = {nt: g.nodes[nt].data["h"] for nt in C.NODE_TYPES}

    with torch.no_grad():
        embs = model.get_final_embeddings(g, fd)

    def to_df(idx_to_id: Dict[int, str], emb: torch.Tensor,
              id_col: str) -> pd.DataFrame:
        arr = emb.detach().cpu().numpy()
        df = pd.DataFrame(arr, columns=[f"emb_{i}" for i in range(arr.shape[1])])
        df.insert(0, id_col, [idx_to_id[i] for i in range(arr.shape[0])])
        return df

    prov_df = to_df(artifacts.prov_idx_to_id, embs[C.PROV_NTYPE], "ProvID")
    unit_df = to_df(artifacts.unit_idx_to_id, embs[C.UNIT_NTYPE], "DepartmentID")
    enc_df  = to_df(artifacts.enc_idx_to_id,  embs[C.ENC_NTYPE],  "LogID")

    # Type-average embeddings for cold-start fallback. Need the original
    # attrs dataframes so we can group by ProviderType / UnitType.
    if raw_prov_attrs is not None and "ProviderType" in raw_prov_attrs.columns:
        joined = prov_df.merge(raw_prov_attrs[["ProvID", "ProviderType"]],
                               on="ProvID", how="left")
        prov_type_avg = (joined
                         .drop(columns=["ProvID"])
                         .groupby("ProviderType")
                         .mean()
                         .reset_index())
    else:
        prov_type_avg = pd.DataFrame()

    if raw_unit_attrs is not None and "UnitType" in raw_unit_attrs.columns:
        joined = unit_df.merge(raw_unit_attrs[["DepartmentID", "UnitType"]],
                               on="DepartmentID", how="left")
        unit_type_avg = (joined
                         .drop(columns=["DepartmentID"])
                         .groupby("UnitType")
                         .mean()
                         .reset_index())
    else:
        unit_type_avg = pd.DataFrame()

    return EmbeddingTables(
        provider=prov_df, unit=unit_df, encounter=enc_df,
        provider_type_avg=prov_type_avg, unit_type_avg=unit_type_avg,
    )


# =====================================================================
def save_embeddings(tables: EmbeddingTables, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables.provider.to_parquet(out_dir / "surgeon_embeddings.parquet", index=False)
    tables.unit.to_parquet(out_dir / "unit_embeddings.parquet",       index=False)
    tables.encounter.to_parquet(out_dir / "encounter_embeddings.parquet", index=False)
    if not tables.provider_type_avg.empty:
        tables.provider_type_avg.to_parquet(
            out_dir / "provider_type_avg_embeddings.parquet", index=False)
    if not tables.unit_type_avg.empty:
        tables.unit_type_avg.to_parquet(
            out_dir / "unit_type_avg_embeddings.parquet", index=False)


# =====================================================================
# Inference-time helpers - identical signatures to what the deployed
# PeriOp_NSQIP_model.py will call after wiring in the SQL Part B
# results. Keeping these here lets us unit-test the lookup logic.
# =====================================================================
def aggregate_provider_embedding(
    prov_ids: pd.Series,
    prov_emb_table: pd.DataFrame,
    type_avg_table: pd.DataFrame,
    provider_types: Optional[pd.Series] = None,
) -> np.ndarray:
    """Mean of trained surgeon embeddings for the providers on this
    case. Unknown IDs fall back to the type-average from training."""
    cols = [c for c in prov_emb_table.columns if c.startswith("emb_")]
    # Dedupe defensively - duplicate ProvIDs in the embedding table
    # would make lut.loc[pid] return a DataFrame instead of a Series.
    lut  = (prov_emb_table.drop_duplicates(subset="ProvID")
                          .set_index("ProvID")[cols])
    type_lut = (type_avg_table.drop_duplicates(subset="ProviderType")
                              .set_index("ProviderType")[cols]
                if not type_avg_table.empty else None)

    rows = []
    for i, pid in enumerate(prov_ids):
        if pid in lut.index:
            rows.append(lut.loc[pid].values)
        elif (type_lut is not None and provider_types is not None
              and provider_types.iloc[i] in type_lut.index):
            rows.append(type_lut.loc[provider_types.iloc[i]].values)
    if not rows:
        return np.zeros(len(cols), dtype=np.float32)
    return np.mean(np.vstack(rows), axis=0).astype(np.float32)


def aggregate_unit_embedding(
    dept_ids: pd.Series,
    hours:    pd.Series,
    unit_emb_table: pd.DataFrame,
    type_avg_table: pd.DataFrame,
    unit_types: Optional[pd.Series] = None,
) -> np.ndarray:
    """Hours-weighted average of trained unit embeddings for this
    encounter's trajectory."""
    cols = [c for c in unit_emb_table.columns if c.startswith("emb_")]
    lut  = (unit_emb_table.drop_duplicates(subset="DepartmentID")
                          .set_index("DepartmentID")[cols])
    type_lut = (type_avg_table.drop_duplicates(subset="UnitType")
                              .set_index("UnitType")[cols]
                if not type_avg_table.empty else None)

    rows, ws = [], []
    for i, did in enumerate(dept_ids):
        h = float(hours.iloc[i]) if pd.notna(hours.iloc[i]) else 0.0
        if h <= 0:
            continue
        if did in lut.index:
            rows.append(lut.loc[did].values); ws.append(h)
        elif (type_lut is not None and unit_types is not None
              and unit_types.iloc[i] in type_lut.index):
            rows.append(type_lut.loc[unit_types.iloc[i]].values); ws.append(h)
    if not rows:
        return np.zeros(len(cols), dtype=np.float32)
    weights = np.asarray(ws, dtype=np.float32)
    weights = weights / weights.sum()
    return (np.vstack(rows) * weights[:, None]).sum(axis=0).astype(np.float32)
