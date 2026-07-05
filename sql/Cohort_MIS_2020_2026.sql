USE Clarity;
/* =====================================================================
   Cohort_MIS_2020_2026.sql
   ---------------------------------------------------------------------
   Adult (>=18) MIS (laparoscopic or robotic) general-surgery cases
   in 2022-2026, stratified random sample up to 50,000 cases.

   CPT source: CLARITY_TDL_TRAN (billing transactions) - matches the
   pattern used in PeriOp_NSQIP_Query_CTE_corrected.sql. The primary
   CPT for an encounter is the row with the highest RVU_WORK.

   Inclusion criteria:
     * SURGERY_DATE between 2022-01-01 and 2026-12-31
     * Completed surgery (STATUS_C = 2)
     * Not cancelled / not procedure-not-performed
     * Has inpatient encounter linkage (PAT_OR_ADM_LINK)
     * Has a billing CPT in the 47-code MIS general-surgery list
     * Patient age >= 18 at surgery

   Sampling:
     * Stratified by SurgeryYear (5 years -> ~10,000 cases each)
     * Pseudo-random ordering via NEWID()

   Output columns (one row per case):
     LogID, EncounterCSN, PAT_ID, SurgeryDate, SurgeryYear,
     PrimaryCPT, AgeYears

   Save the result as `cohort_mis_2020_2026.csv`.
   ===================================================================== */


IF OBJECT_ID('tempdb..#CohortLogIDs') IS NOT NULL DROP TABLE #CohortLogIDs;

-- ---------------------------------------------------------------------
-- WCM campus filter. Set @WCM_ParentLocID to the LOC_ID of the WCM
-- hospital parent location, found via the discovery query:
--   SELECT ploc.LOC_ID, ploc.LOC_NAME, COUNT(*)
--   FROM OR_LOG orl
--   JOIN CLARITY_LOC loc  ON orl.LOC_ID = loc.LOC_ID
--   LEFT JOIN CLARITY_LOC ploc ON loc.HOSP_PARENT_LOC_ID = ploc.LOC_ID
--   GROUP BY ploc.LOC_ID, ploc.LOC_NAME ORDER BY 3 DESC;
-- ---------------------------------------------------------------------
DECLARE @WCM_ParentLocID INT = 10100001;   -- WCM hospital parent LOC_ID

-- ---------------------------------------------------------------------
-- All cases meeting the inclusion criteria, before sampling.
-- ---------------------------------------------------------------------
WITH OR_Cases AS (
    SELECT
        LogID         = CONVERT(varchar, orl.LOG_ID)
       ,EncounterCSN  = COALESCE(poal.OR_LINK_CSN, poal.PAT_ENC_CSN_ID)
       ,orl.PAT_ID
       ,SurgeryDate   = CONVERT(date, orl.SURGERY_DATE)
       ,SurgeryYear   = YEAR(orl.SURGERY_DATE)
       ,AgeYears      = CONVERT(decimal(5, 2),
                                DATEDIFF(DAY, p.BIRTH_DATE, orl.SURGERY_DATE) / 365.25)
    FROM OR_LOG orl
    LEFT JOIN OR_LOG_2 orl2     ON orl.LOG_ID = orl2.LOG_ID
    INNER JOIN PATIENT p        ON orl.PAT_ID = p.PAT_ID
    INNER JOIN PAT_OR_ADM_LINK poal ON orl.LOG_ID = poal.LOG_ID
    INNER JOIN CLARITY_LOC loc  ON orl.LOC_ID = loc.LOC_ID  -- OR location
    WHERE orl.SURGERY_DATE BETWEEN '2022-01-01' AND '2026-12-31'
      AND orl.STATUS_C             = 2
      AND orl2.PROC_NOT_DONE_RSN_C IS NULL
      AND orl.PROC_NOT_PERF_C      IS NULL
      AND p.BIRTH_DATE             IS NOT NULL
      AND DATEDIFF(YEAR, p.BIRTH_DATE, orl.SURGERY_DATE) >= 18
      AND COALESCE(poal.OR_LINK_CSN, poal.PAT_ENC_CSN_ID) IS NOT NULL
      -- Limit to ORs done at WCM (campus = HOSP_PARENT_LOC_ID rollup).
      AND loc.HOSP_PARENT_LOC_ID = @WCM_ParentLocID
),
-- Pull the highest-RVU CPT per encounter from CLARITY_TDL_TRAN,
-- restricted to the 47 MIS CPTs we care about.
CPT AS (
    SELECT
        trn.PAT_ENC_CSN_ID
       ,trn.INT_PAT_ID
       ,trn.CPT_CODE
       ,trn.RVU_WORK
       ,SEQ = ROW_NUMBER() OVER (
                PARTITION BY trn.PAT_ENC_CSN_ID
                ORDER BY trn.RVU_WORK DESC
              )
    FROM CLARITY_TDL_TRAN trn
    INNER JOIN OR_Cases o
        ON  trn.INT_PAT_ID     = o.PAT_ID
        AND trn.PAT_ENC_CSN_ID = o.EncounterCSN
    WHERE trn.CPT_CODE IN (
        '44204','49650','44970','45400','43775','44202','47563','47562',
        '43659','44180','44187','43280','43644','49652','44206','44207',
        '44615','49000','43771','49654','44213','38120','44188','49321',
        '45397','44227','43281','44899','43289','43279','49329','38570',
        '44238','43620','43770','49659','47379','38129','43645','43284',
        '48999','49322','49324','38571','43499','47570','49323'
    )
),
-- Inner-join the cohort to its primary (top-RVU) MIS CPT.
Candidates AS (
    SELECT
        o.LogID
       ,o.EncounterCSN
       ,o.PAT_ID
       ,o.SurgeryDate
       ,o.SurgeryYear
       ,PrimaryCPT = c.CPT_CODE
       ,o.AgeYears
    FROM OR_Cases o
    INNER JOIN CPT c
        ON  c.PAT_ENC_CSN_ID = o.EncounterCSN
        AND c.SEQ            = 1
),
-- Stratified random sample by SurgeryYear.
Sampled AS (
    SELECT
        *
       ,RowInYear = ROW_NUMBER() OVER (
                        PARTITION BY SurgeryYear
                        ORDER BY ABS(CHECKSUM(NEWID()))
                    )
    FROM Candidates
)
SELECT
    LogID
   ,EncounterCSN
   ,PAT_ID
   ,SurgeryDate
   ,SurgeryYear
   ,PrimaryCPT
   ,AgeYears
INTO #CohortLogIDs
FROM Sampled
WHERE RowInYear <= 10000;   -- ~50,000 total over 5 years (2022-2026)

CREATE NONCLUSTERED INDEX IX_CohortLogIDs_LogID ON #CohortLogIDs (LogID);
CREATE NONCLUSTERED INDEX IX_CohortLogIDs_CSN   ON #CohortLogIDs (EncounterCSN);

-- Return the cohort for export to CSV.
SELECT * FROM #CohortLogIDs ORDER BY SurgeryYear, SurgeryDate;


-- ---------------------------------------------------------------------
-- Optional sanity-check queries (run after the main SELECT to QA the
-- cohort). Comment-only - won't execute as part of the file.
-- ---------------------------------------------------------------------
/*
-- Total candidate count (before sampling)
WITH OR_Cases AS (...), CPT AS (...), Candidates AS (...)
SELECT COUNT(*) AS n_candidates FROM Candidates;

-- Year breakdown post-sample
WITH OR_Cases AS (...), CPT AS (...), Candidates AS (...), Sampled AS (...)
SELECT SurgeryYear, n_in_year = COUNT(*) FROM Sampled
WHERE RowInYear <= 10000 GROUP BY SurgeryYear ORDER BY SurgeryYear;

-- CPT distribution post-sample
WITH OR_Cases AS (...), CPT AS (...), Candidates AS (...), Sampled AS (...)
SELECT PrimaryCPT, n = COUNT(*) FROM Sampled WHERE RowInYear <= 10000
GROUP BY PrimaryCPT ORDER BY n DESC;
*/
