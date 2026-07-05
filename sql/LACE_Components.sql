USE Clarity;
/* =====================================================================
   LACE_Components.sql
   ---------------------------------------------------------------------
   Pulls the two LACE components that are NOT in the current bulk-features
   extract:
     - C: index-encounter Charlson comorbidity from ICD-10 codes
     - E: emergency-department visits in the 180 days prior to the index
          surgery encounter
   These join to the cohort keys already produced by
   Load_CohortTempTable.sql (#CohortLogIDs), so the outputs align 1:1
   with the WCM MIS cohort used everywhere else.

   L (length of stay) and A (emergent admission) are already derivable
   from A3 (unit trajectory) — see analysis/lace_baseline.py.

   Deliverables:
     - lace_charlson.csv   one row per LogID: LogID + Charlson score
                           + 17 individual comorbidity flags
     - lace_ed_visits.csv  one row per LogID: LogID + n_ed_visits_180d

   Assumes: #CohortLogIDs already loaded (LogID -> PAT_ID + surgery date).
   Run after Load_CohortTempTable.sql in the same session.
   ===================================================================== */


/* ---------------------------------------------------------------------
   Charlson comorbidity — Deyo/Sundararajan ICD-10 mapping.
   Weighted sum over 17 categories; capped at 5 for the C component of
   LACE (van Walraven 2010 §methods).

   Approach: for each index encounter, look up all diagnoses with an
   ENC_DX / DX_LIST or PAT_ENC_DX row on or before the surgery date and
   map ICD-10 to Charlson categories via case statement.
   --------------------------------------------------------------------- */
IF OBJECT_ID('tempdb..#LACE_Charlson_Raw') IS NOT NULL DROP TABLE #LACE_Charlson_Raw;

SELECT  c.LogID,
        dxs.icd10 AS icd10,
        CASE
          -- Weight 1
          WHEN dxs.icd10 LIKE 'I21%' OR dxs.icd10 LIKE 'I22%'                                       THEN 'MI'
          WHEN dxs.icd10 LIKE 'I50%'                                                                    THEN 'CHF'
          WHEN dxs.icd10 LIKE 'I70%' OR dxs.icd10 LIKE 'I71%' OR dxs.icd10 LIKE 'I73%'
               OR dxs.icd10 LIKE 'I74%'                                                                 THEN 'PVD'
          WHEN dxs.icd10 LIKE 'I6[0-9]%' OR dxs.icd10 LIKE 'G45%' OR dxs.icd10 LIKE 'G46%'       THEN 'CVD'
          WHEN dxs.icd10 LIKE 'F00%' OR dxs.icd10 LIKE 'F01%' OR dxs.icd10 LIKE 'F02%'
               OR dxs.icd10 LIKE 'F03%' OR dxs.icd10 LIKE 'G30%'                                     THEN 'DEMENTIA'
          WHEN dxs.icd10 LIKE 'J4[0-7]%' OR dxs.icd10 LIKE 'J6[0-7]%'                                THEN 'COPD'
          WHEN dxs.icd10 LIKE 'M05%' OR dxs.icd10 LIKE 'M06%'                                       THEN 'RHEUM'
          WHEN dxs.icd10 LIKE 'K25%' OR dxs.icd10 LIKE 'K26%' OR dxs.icd10 LIKE 'K27%'
               OR dxs.icd10 LIKE 'K28%'                                                                 THEN 'PUD'
          WHEN dxs.icd10 LIKE 'B18%' OR dxs.icd10 LIKE 'K70[023]%' OR dxs.icd10 LIKE 'K71%'
               OR dxs.icd10 LIKE 'K73%' OR dxs.icd10 LIKE 'K74%'                                     THEN 'LIVER_MILD'
          WHEN dxs.icd10 LIKE 'E10%' OR dxs.icd10 LIKE 'E11%' OR dxs.icd10 LIKE 'E12%'
               OR dxs.icd10 LIKE 'E13%' OR dxs.icd10 LIKE 'E14%'                                     THEN 'DM_UNC'
          -- Weight 2
          WHEN dxs.icd10 LIKE 'G81%' OR dxs.icd10 LIKE 'G82%'                                       THEN 'HEMI'
          WHEN dxs.icd10 LIKE 'N03[234567]%' OR dxs.icd10 LIKE 'N05[234567]%'
               OR dxs.icd10 LIKE 'N18%' OR dxs.icd10 LIKE 'N19%'                                     THEN 'RENAL'
          -- (DM with complications from more specific E1x.2-8)
          WHEN dxs.icd10 LIKE 'E1[01234][23456789]%'                                                    THEN 'DM_COMP'
          WHEN dxs.icd10 LIKE 'C[0-7][0-9]%' OR dxs.icd10 LIKE 'C8[01234]%'
               OR dxs.icd10 LIKE 'C88%' OR dxs.icd10 LIKE 'C9[0-6]%'                                 THEN 'CANCER'
          -- Weight 3
          WHEN dxs.icd10 LIKE 'K70[469]%' OR dxs.icd10 LIKE 'K72%' OR dxs.icd10 LIKE 'K76[567]%' THEN 'LIVER_SEV'
          -- Weight 6
          WHEN dxs.icd10 LIKE 'C77%' OR dxs.icd10 LIKE 'C78%' OR dxs.icd10 LIKE 'C79%'
               OR dxs.icd10 LIKE 'C80%'                                                                 THEN 'METS'
          WHEN dxs.icd10 LIKE 'B2[0-4]%'                                                                THEN 'AIDS'
          ELSE NULL
        END AS charlson_cat
INTO   #LACE_Charlson_Raw
FROM   #CohortLogIDs c
CROSS  APPLY (
    -- Dx over the 24 months before surgery: active problem list UNION
    -- encounter diagnoses. Mirrors Bulk_Features_From_Cohort.sql DxHistory;
    -- the old index-encounter-only join under-captured chronic comorbidity.
    SELECT DISTINCT icd10 = ed.CODE
    FROM (
        SELECT pl.DX_ID
        FROM PROBLEM_LIST pl
        WHERE pl.PAT_ID = c.PAT_ID
          AND pl.NOTED_DATE <= c.SurgeryDate
          AND (pl.RESOLVED_DATE IS NULL OR pl.RESOLVED_DATE >= c.SurgeryDate)
        UNION
        SELECT pedx.DX_ID
        FROM PAT_ENC penc
        INNER JOIN PAT_ENC_DX pedx ON pedx.PAT_ENC_CSN_ID = penc.PAT_ENC_CSN_ID
        WHERE penc.PAT_ID = c.PAT_ID
          AND penc.CONTACT_DATE >= DATEADD(MONTH, -24, c.SurgeryDate)
          AND penc.CONTACT_DATE <= c.SurgeryDate
    ) d
    INNER JOIN EDG_CURRENT_ICD10 ed ON ed.DX_ID = d.DX_ID
    WHERE ed.CODE IS NOT NULL
) dxs;

-- pivot to per-LogID flags + Charlson score
IF OBJECT_ID('tempdb..#LACE_Charlson') IS NOT NULL DROP TABLE #LACE_Charlson;

SELECT  LogID,
        MAX(CASE WHEN charlson_cat = 'MI'         THEN 1 ELSE 0 END) AS mi,
        MAX(CASE WHEN charlson_cat = 'CHF'        THEN 1 ELSE 0 END) AS chf,
        MAX(CASE WHEN charlson_cat = 'PVD'        THEN 1 ELSE 0 END) AS pvd,
        MAX(CASE WHEN charlson_cat = 'CVD'        THEN 1 ELSE 0 END) AS cvd,
        MAX(CASE WHEN charlson_cat = 'DEMENTIA'   THEN 1 ELSE 0 END) AS dementia,
        MAX(CASE WHEN charlson_cat = 'COPD'       THEN 1 ELSE 0 END) AS copd,
        MAX(CASE WHEN charlson_cat = 'RHEUM'      THEN 1 ELSE 0 END) AS rheum,
        MAX(CASE WHEN charlson_cat = 'PUD'        THEN 1 ELSE 0 END) AS pud,
        MAX(CASE WHEN charlson_cat = 'LIVER_MILD' THEN 1 ELSE 0 END) AS liver_mild,
        MAX(CASE WHEN charlson_cat = 'DM_UNC'     THEN 1 ELSE 0 END) AS dm_uncomp,
        MAX(CASE WHEN charlson_cat = 'HEMI'       THEN 1 ELSE 0 END) AS hemiplegia,
        MAX(CASE WHEN charlson_cat = 'RENAL'      THEN 1 ELSE 0 END) AS renal,
        MAX(CASE WHEN charlson_cat = 'DM_COMP'    THEN 1 ELSE 0 END) AS dm_comp,
        MAX(CASE WHEN charlson_cat = 'CANCER'     THEN 1 ELSE 0 END) AS cancer,
        MAX(CASE WHEN charlson_cat = 'LIVER_SEV'  THEN 1 ELSE 0 END) AS liver_sev,
        MAX(CASE WHEN charlson_cat = 'METS'       THEN 1 ELSE 0 END) AS mets,
        MAX(CASE WHEN charlson_cat = 'AIDS'       THEN 1 ELSE 0 END) AS aids
INTO   #LACE_Charlson
FROM   #LACE_Charlson_Raw
GROUP  BY LogID;

/* ---------------------------------------------------------------------
   E — Emergency-department visits in the 180 days BEFORE the index
   surgery date.

   Counts ACTUAL ED visits from F_ED_ENCOUNTERS (one row per ED visit,
   including treat-and-release). The old version counted prior emergency
   *admissions* (PAT_ENC_HSP.HOSP_ADMSN_TYPE_C IN (3,7)) -- that misses the
   majority of ED visits (those discharged home) and badly undercounts E.
   --------------------------------------------------------------------- */
IF OBJECT_ID('tempdb..#LACE_EDVisits') IS NOT NULL DROP TABLE #LACE_EDVisits;

SELECT  c.LogID,
        COUNT(DISTINCT fed.PAT_ENC_CSN_ID) AS n_ed_visits_180d
INTO   #LACE_EDVisits
FROM   #CohortLogIDs c
LEFT   JOIN F_ED_ENCOUNTERS fed          -- actual ED visits (incl. treat-and-release)
         ON  fed.PAT_ID = c.PAT_ID        -- VERIFY: F_ED_ENCOUNTERS.PAT_ID exists at your
                                          -- site; if not, join PAT_ENC on PAT_ENC_CSN_ID for PAT_ID
        AND fed.ADT_ARRIVAL_DTTM <  c.SurgeryDate
        AND fed.ADT_ARRIVAL_DTTM >= DATEADD(day, -180, c.SurgeryDate)
GROUP  BY c.LogID;

/* ---------------------------------------------------------------------
   T — Hospital admissions in the 365 days BEFORE the index surgery
   date. Component of the HOSPITAL readmission score (Donzé 2013).

   Counts distinct PAT_ENC_HSP encounters (any admission type, not
   just ED). Used by analysis/hospital_score.py to bin as:
     0-1 admissions -> 0 pts, 2-5 -> 2 pts, >5 -> 5 pts.
   --------------------------------------------------------------------- */
IF OBJECT_ID('tempdb..#HOSPITAL_Admits365') IS NOT NULL DROP TABLE #HOSPITAL_Admits365;

SELECT  c.LogID,
        COUNT(DISTINCT prior_hsp.PAT_ENC_CSN_ID) AS n_admits_365d
INTO   #HOSPITAL_Admits365
FROM   #CohortLogIDs c
LEFT   JOIN PAT_ENC_HSP prior_hsp
         ON  prior_hsp.PAT_ID          = c.PAT_ID
        AND prior_hsp.PAT_ENC_CSN_ID  <> c.EncounterCSN        -- exclude the index admission
        AND prior_hsp.HOSP_ADMSN_TIME IS NOT NULL              -- inpatient admissions only
        AND prior_hsp.HOSP_ADMSN_TIME <  c.SurgeryDate
        AND prior_hsp.HOSP_ADMSN_TIME >= DATEADD(day, -365, c.SurgeryDate)
GROUP  BY c.LogID;

/* ---------------------------------------------------------------------
   Final SELECT — one row per cohort LogID with all LACE components
   + HOSPITAL-T (prior-year admissions count).
   The Python side (analysis/lace_baseline.py, analysis/hospital_score.py)
   applies the score-specific lookups.
   --------------------------------------------------------------------- */
SELECT  c.LogID,
        COALESCE(ch.mi, 0)         AS mi,
        COALESCE(ch.chf, 0)        AS chf,
        COALESCE(ch.pvd, 0)        AS pvd,
        COALESCE(ch.cvd, 0)        AS cvd,
        COALESCE(ch.dementia, 0)   AS dementia,
        COALESCE(ch.copd, 0)       AS copd,
        COALESCE(ch.rheum, 0)      AS rheum,
        COALESCE(ch.pud, 0)        AS pud,
        COALESCE(ch.liver_mild, 0) AS liver_mild,
        COALESCE(ch.dm_uncomp, 0)  AS dm_uncomp,
        COALESCE(ch.hemiplegia, 0) AS hemiplegia,
        COALESCE(ch.renal, 0)      AS renal,
        COALESCE(ch.dm_comp, 0)    AS dm_comp,
        COALESCE(ch.cancer, 0)     AS cancer,
        COALESCE(ch.liver_sev, 0)  AS liver_sev,
        COALESCE(ch.mets, 0)       AS mets,
        COALESCE(ch.aids, 0)       AS aids,
        COALESCE(ed.n_ed_visits_180d, 0) AS n_ed_visits_180d,
        COALESCE(hs.n_admits_365d,   0) AS n_admits_365d
FROM   #CohortLogIDs c
LEFT   JOIN #LACE_Charlson       ch ON ch.LogID = c.LogID
LEFT   JOIN #LACE_EDVisits       ed ON ed.LogID = c.LogID
LEFT   JOIN #HOSPITAL_Admits365  hs ON hs.LogID = c.LogID
ORDER  BY c.LogID;

-- cleanup
DROP TABLE IF EXISTS #LACE_Charlson_Raw;
DROP TABLE IF EXISTS #LACE_Charlson;
DROP TABLE IF EXISTS #LACE_EDVisits;
DROP TABLE IF EXISTS #HOSPITAL_Admits365;
