"""
Inference-time integration helpers for PeriOp_NSQIP_model.py.

After MedHG-PS is trained and embeddings are saved (extract_embeddings.py),
this module turns the two per-MRN SQL Part B results into a single
embedding vector that gets concatenated to the RF tabular feature vector.

Wire it into PeriOp_NSQIP_model.py like this:

    from medhg_ps.inference_hook import GnnEmbeddings, gen_providers_query, \\
        gen_unit_trajectory_query

    # Module-level (cache singleton):
    _gnn_emb = GnnEmbeddings.load(Path("resources") / "gnn_embeddings")

    # In _run_all_queries, after the existing 13 queries:
    prov_df = execute_query(gen_providers_query(mrn))
    unit_df = execute_query(gen_unit_trajectory_query(mrn))

    # In _build_per_case_rows, after the per-case dict is built:
    prov_vec, unit_vec = _gnn_emb.lookup_for_case(
        log_id=row["LogID"],
        prov_rows=prov_df[prov_df["LogID"] == row["LogID"]],
        unit_rows=unit_df[unit_df["LogID"] == row["LogID"]],
    )
    d["_gnn_prov_emb"] = prov_vec
    d["_gnn_unit_emb"] = unit_vec

The retrained RF then takes [tabular | prov_emb | unit_emb] as its
input feature vector. The expansion of fill_values / feature_names /
categorical_columns in the pickle is handled in Surgery_shortmodel.ipynb
when retraining.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .extract_embeddings import (
    aggregate_provider_embedding,
    aggregate_unit_embedding,
)


# =====================================================================
# Loaded embedding tables
# =====================================================================
@dataclass
class GnnEmbeddings:
    provider:          pd.DataFrame
    unit:              pd.DataFrame
    provider_type_avg: pd.DataFrame
    unit_type_avg:     pd.DataFrame

    @classmethod
    def load(cls, embed_dir: Path) -> "GnnEmbeddings":
        embed_dir = Path(embed_dir)

        def _opt(name: str) -> pd.DataFrame:
            p = embed_dir / name
            return pd.read_parquet(p) if p.exists() else pd.DataFrame()

        return cls(
            provider          = pd.read_parquet(embed_dir / "surgeon_embeddings.parquet"),
            unit              = pd.read_parquet(embed_dir / "unit_embeddings.parquet"),
            provider_type_avg = _opt("provider_type_avg_embeddings.parquet"),
            unit_type_avg     = _opt("unit_type_avg_embeddings.parquet"),
        )

    @property
    def prov_dim(self) -> int:
        return sum(1 for c in self.provider.columns if c.startswith("emb_"))

    @property
    def unit_dim(self) -> int:
        return sum(1 for c in self.unit.columns if c.startswith("emb_"))

    # -----------------------------------------------------------------
    def lookup_for_case(
        self,
        log_id: str,
        prov_rows: pd.DataFrame,
        unit_rows: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Aggregate embeddings for one OR case.

        prov_rows: subset of B1 SQL output for this LogID
                   (cols: ProvID, ProviderType).
        unit_rows: subset of B2 SQL output for this LogID
                   (cols: DepartmentID, UnitType, Hours).
        """
        prov_vec = aggregate_provider_embedding(
            prov_ids=prov_rows["ProvID"].astype(str),
            prov_emb_table=self.provider.assign(
                ProvID=self.provider["ProvID"].astype(str)),
            type_avg_table=self.provider_type_avg,
            provider_types=prov_rows.get("ProviderType"),
        )
        unit_vec = aggregate_unit_embedding(
            dept_ids=unit_rows["DepartmentID"].astype(str),
            hours=unit_rows["Hours"].astype(float),
            unit_emb_table=self.unit.assign(
                DepartmentID=self.unit["DepartmentID"].astype(str)),
            type_avg_table=self.unit_type_avg,
            unit_types=unit_rows.get("UnitType"),
        )
        return prov_vec, unit_vec


# =====================================================================
# SQL templates - Part B from GNN_Graph_Extract.sql, ready to call from
# PeriOp_NSQIP_model.py's existing query-builder pattern.
# =====================================================================
_PROVIDERS_SQL = """
WITH PatCases AS (
    SELECT orl.LOG_ID, orl.PAT_ID, orl.SURGERY_DATE
    FROM OR_LOG orl
    INNER JOIN PATIENT p ON orl.PAT_ID = p.PAT_ID
    LEFT JOIN OR_LOG_2 orl2 ON orl.LOG_ID = orl2.LOG_ID
    WHERE p.PAT_MRN_ID = '{mrn}'
      AND orl2.PROC_NOT_DONE_RSN_C IS NULL
),
ProviderTypeRule AS (
    SELECT
        ser.PROV_ID
       ,IsCRNA       = CASE WHEN ser.EMPLOYED_CRNA_YN = '1' THEN 1 ELSE 0 END
       ,IsResident   = CASE WHEN ser.IS_RESIDENT     = 'Y' THEN 1 ELSE 0 END
       ,IsPhysician  = CASE WHEN ser.DOCTORS_DEGREE LIKE '%MD%'
                              OR ser.DOCTORS_DEGREE LIKE '%DO%' THEN 1 ELSE 0 END
       ,IsNurseLike  = CASE WHEN ser.MCD_PROF_CD_C = 22 THEN 1 ELSE 0 END
       ,IsTechLike   = CASE WHEN ser.MCD_PROF_CD_C IN (52, 53) THEN 1 ELSE 0 END
    FROM CLARITY_SER ser
)
SELECT
    LogID         = CONVERT(varchar, olas.LOG_ID)
   ,ProvID        = CONVERT(varchar, olas.SURG_ID)
   ,RoleCode      = olas.ROLE_C
   ,RoleSource    = 'OR_LOG_ALL_SURG'
   ,ProviderType  =
        CASE
            WHEN ptr.IsResident  = 1 THEN 'SurgicalTeam'
            WHEN ptr.IsCRNA      = 1 THEN 'OtherClinician'
            WHEN ptr.IsPhysician = 1 THEN 'SurgicalTeam'
            WHEN ptr.IsNurseLike = 1 THEN 'Nurse'
            WHEN ptr.IsTechLike  = 1 THEN 'Technician'
            ELSE 'Other'
        END
FROM OR_LOG_ALL_SURG olas
INNER JOIN PatCases pc ON olas.LOG_ID = pc.LOG_ID
LEFT JOIN ProviderTypeRule ptr ON olas.SURG_ID = ptr.PROV_ID

UNION ALL

SELECT
    LogID         = CONVERT(varchar, pc.LOG_ID)
   ,ProvID        = CONVERT(varchar, anes.RESPONSIBLE_PROV_ID)
   ,RoleCode      = NULL
   ,RoleSource    = 'DM_ANESTHESIA.RESPONSIBLE_PROV_ID'
   ,ProviderType  = 'OtherClinician'
FROM PatCases pc
INNER JOIN DM_ANESTHESIA anes
    ON  anes.PAT_ID = pc.PAT_ID
    AND CONVERT(date, anes.RECORD_DATE) = CONVERT(date, pc.SURGERY_DATE)
WHERE anes.RESPONSIBLE_PROV_ID IS NOT NULL

UNION ALL

SELECT
    LogID         = CONVERT(varchar, pc.LOG_ID)
   ,ProvID        = LTRIM(RTRIM(split.value))
   ,RoleCode      = NULL
   ,RoleSource    = 'DM_ANESTHESIA.ALL_AN_STAFF'
   ,ProviderType  =
        CASE
            WHEN ptr.IsCRNA      = 1 THEN 'OtherClinician'
            WHEN ptr.IsResident  = 1 THEN 'OtherClinician'
            WHEN ptr.IsPhysician = 1 THEN 'OtherClinician'
            WHEN ptr.IsNurseLike = 1 THEN 'Nurse'
            ELSE 'OtherClinician'
        END
FROM PatCases pc
INNER JOIN DM_ANESTHESIA anes
    ON  anes.PAT_ID = pc.PAT_ID
    AND CONVERT(date, anes.RECORD_DATE) = CONVERT(date, pc.SURGERY_DATE)
CROSS APPLY STRING_SPLIT(anes.ALL_AN_STAFF, ';') split
LEFT JOIN ProviderTypeRule ptr
    ON LTRIM(RTRIM(split.value)) = ptr.PROV_ID
WHERE anes.ALL_AN_STAFF IS NOT NULL
  AND LEN(LTRIM(RTRIM(split.value))) > 0
"""


_UNIT_TRAJECTORY_SQL = """
WITH PatCases AS (
    SELECT orl.LOG_ID, EncounterCSN = poal.PAT_ENC_CSN_ID
    FROM OR_LOG orl
    INNER JOIN PATIENT p ON orl.PAT_ID = p.PAT_ID
    LEFT JOIN PAT_OR_ADM_LINK poal ON orl.LOG_ID = poal.LOG_ID
    WHERE p.PAT_MRN_ID = '{mrn}'
      AND poal.PAT_ENC_CSN_ID IS NOT NULL
)
SELECT
    LogID          = CONVERT(varchar, pc.LOG_ID)
   ,EncounterCSN   = adt.PAT_ENC_CSN_ID
   ,DepartmentID   = CONVERT(varchar, adt.DEPARTMENT_ID)
   ,DepartmentName = dep.DEPARTMENT_NAME
   ,UnitType       =
        CASE
            WHEN dep.OR_UNIT_TYPE_C IN (1, 2)             THEN 'OR'
            WHEN dep.OR_UNIT_TYPE_C IN (3, 4)             THEN 'Intermediate'
            WHEN dep.DEPARTMENT_NAME LIKE '%ICU%'         THEN 'Intensive'
            WHEN dep.DEPARTMENT_NAME LIKE '%CCU%'         THEN 'Intensive'
            WHEN dep.DEPARTMENT_NAME LIKE '%INTENSIVE%'   THEN 'Intensive'
            WHEN dep.DEPARTMENT_NAME LIKE '%CRITICAL%'    THEN 'Intensive'
            WHEN dep.DEPARTMENT_NAME LIKE '%STEP DOWN%'   THEN 'Intermediate'
            WHEN dep.DEPARTMENT_NAME LIKE '%STEPDOWN%'    THEN 'Intermediate'
            WHEN dep.DEPARTMENT_NAME LIKE '%IMU%'         THEN 'Intermediate'
            WHEN dep.DEPARTMENT_NAME LIKE '%INTERMEDIATE%' THEN 'Intermediate'
            WHEN dep.DEPARTMENT_NAME LIKE '%TELEMETRY%'   THEN 'Intermediate'
            WHEN dep.INPATIENT_DEPT_YN = 'Y'              THEN 'Acute'
            WHEN dep.ADT_UNIT_TYPE_C = 0                  THEN 'Acute'
            ELSE 'Other'
        END
   ,InTime         = adt.EFFECTIVE_TIME
   ,OutTime        = LEAD(adt.EFFECTIVE_TIME) OVER (
                          PARTITION BY adt.PAT_ENC_CSN_ID
                          ORDER BY adt.EFFECTIVE_TIME)
   ,Hours          = DATEDIFF(MINUTE, adt.EFFECTIVE_TIME,
                              LEAD(adt.EFFECTIVE_TIME) OVER (
                                PARTITION BY adt.PAT_ENC_CSN_ID
                                ORDER BY adt.EFFECTIVE_TIME)) / 60.0
FROM CLARITY_ADT adt
INNER JOIN PatCases pc ON adt.PAT_ENC_CSN_ID = pc.EncounterCSN
LEFT JOIN CLARITY_DEP dep ON adt.DEPARTMENT_ID = dep.DEPARTMENT_ID
WHERE adt.EVENT_TYPE_C IN (1, 3)
  AND adt.EVENT_SUBTYPE_C != 2
"""


def gen_providers_query(mrn: str) -> str:
    return _PROVIDERS_SQL.replace("{mrn}", mrn)


def gen_unit_trajectory_query(mrn: str) -> str:
    return _UNIT_TRAJECTORY_SQL.replace("{mrn}", mrn)
