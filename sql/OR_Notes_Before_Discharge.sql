/* =====================================================================
   OR_Notes_Before_Discharge.sql
   ---------------------------------------------------------------------
   Pull operative, anesthesia, admission summary, and postop progress
   notes for each cohort case, bounded between surgery start time and
   discharge time. One row per (LogID, note) with full text content
   from the LATEST version of each note.

   Feeds the LLM-explanation layer that produces plain-language
   risk-factor narratives for clinicians at discharge.

   PHI WARNING: output contains free-text clinical notes. Treat as
   highest-sensitivity PHI.

   PREREQUISITES (same Cogito session):
     1. Load_CohortTempTable.sql -- creates #CohortLogIDs
     2. USE Clarity;
   ===================================================================== */

USE Clarity;


-- =====================================================================
-- STEP 1: Cohort + note window.
-- Use #CohortLogIDs.EncounterCSN DIRECTLY -- validated: 85.7% of cases
-- have a target-type note on it. (The old PAT_OR_ADM_LINK re-derivation
-- of a separate "inpatient" CSN was dropping cases; the cohort CSN is
-- where the notes actually live.)
--
-- Window = ADMISSION -> DISCHARGE, from CLARITY_ADT on that CSN. Widening
-- the lower bound from surgery-start to admission recovers ~2.4k pre-op
-- admission summaries the old surgery->discharge window excluded. Upper
-- bound stays at discharge (leakage-safe for a discharge-time model).
-- Cases with no ADT discharge (pure ambulatory) drop out here.
-- =====================================================================
IF OBJECT_ID('tempdb..#CohortNoteScope') IS NOT NULL DROP TABLE #CohortNoteScope;

SELECT
    cl.LogID,
    cl.PAT_ID,
    cl.EncounterCSN,
    SurgeryStart  = CONVERT(datetime, cl.SurgeryDate),   -- for output metrics
    AdmitTime     = adt.AdmitTime,                        -- window lower bound
    DischargeTime = adt.DischargeTime                     -- window upper bound
INTO #CohortNoteScope
FROM #CohortLogIDs cl
CROSS APPLY (
    SELECT
        AdmitTime     = MIN(a.EFFECTIVE_TIME),
        DischargeTime = MAX(CASE WHEN a.EVENT_TYPE_C = 2 THEN a.EFFECTIVE_TIME END)
    FROM CLARITY_ADT a WITH (NOLOCK)
    WHERE a.PAT_ENC_CSN_ID = cl.EncounterCSN
      AND (a.EVENT_SUBTYPE_C IS NULL OR a.EVENT_SUBTYPE_C != 2)
) adt
WHERE adt.DischargeTime IS NOT NULL;

CREATE INDEX IX_CohortNoteScope_CSN ON #CohortNoteScope (EncounterCSN);


-- =====================================================================
-- STEP 2: Find relevant notes for those encounters, time-bounded.
-- Joins HNO_INFO directly by CSN and filters to the WCM IP_NOTE_TYPE_C
-- codes that matter (verified against the site's ZC_NOTE_TYPE_IP; the old
-- 19/25/27/29 were wrong -- 19=ED Provider, 27=Interval H&P, 25/29 absent).
--
--   Operative / procedure : 1000004 Op Note, 1000000 Brief Op Note,
--                           1000012 Post-Procedure, 3 Procedures
--   Admission / H&P       : 4 H&P, 27 Interval H&P, 26 H&P (View-Only)
-- Dropped: Progress Notes (1) + Hospital Course (32) -- redundant with the
-- order sequence (postop course) and discharge-adjacent (leakage). Notes
-- now cover what orders can't: operative detail + preop baseline.
--
-- Excluded: Anesthesia (no text note here -- it's in DM_ANESTHESIA) and
-- Discharge Summary (5) -- authored at discharge => outcome leakage.
-- Looked up via ZC_NOTE_TYPE_IP.TYPE_IP_C (NOT ZC_NOTE_TYPE.INTERNAL_ID
-- which is the outpatient lookup).
-- =====================================================================
IF OBJECT_ID('tempdb..#NoteIDs') IS NOT NULL DROP TABLE #NoteIDs;

SELECT DISTINCT
    cs.LogID
   ,cs.PAT_ID
   ,cs.EncounterCSN
   ,cs.SurgeryStart
   ,cs.DischargeTime
   ,hno.NOTE_ID
   ,NoteTypeCode = hno.IP_NOTE_TYPE_C
   ,NoteTypeName = nt.NAME
   ,NoteCreated  = hno.CREATE_INSTANT_DTTM
   ,AuthorID     = hno.CURRENT_AUTHOR_ID
INTO #NoteIDs
FROM #CohortNoteScope cs
INNER JOIN HNO_INFO hno
    ON  hno.PAT_ENC_CSN_ID      = cs.EncounterCSN
    AND hno.CREATE_INSTANT_DTTM >= cs.AdmitTime        -- admission (catches pre-op admit summary)
    AND hno.CREATE_INSTANT_DTTM <= cs.DischargeTime    -- through discharge (inclusive)
-- IP_NOTE_TYPE_C is varchar at WCM (with some huge custom codes like
-- '3041290019' that overflow int). TRY_CAST to bigint so out-of-range
-- values fall through to NULL and get filtered naturally by the IN.
LEFT JOIN ZC_NOTE_TYPE_IP nt
    ON nt.TYPE_IP_C = TRY_CAST(hno.IP_NOTE_TYPE_C AS bigint)
WHERE TRY_CAST(hno.IP_NOTE_TYPE_C AS bigint) IN (
        1000004, 1000000, 1000012, 3,   -- operative / procedure
        4, 27, 26                         -- admission / H&P
      )   -- Progress Notes (1) + Hospital Course (32) dropped: redundant with
          -- the order sequence and leakage-prone (discharge-adjacent synthesis).
  -- WCM uses UNSIGNED_YN (Y/N flag) instead of an integer NOTE_STATUS_C.
  -- Filter excludes drafts; treats NULL as signed (conservative).
  AND (hno.UNSIGNED_YN IS NULL OR hno.UNSIGNED_YN = 'N');

CREATE INDEX IX_NoteIDs_NoteID ON #NoteIDs (NOTE_ID);


-- =====================================================================
-- STEP 3: Resolve to the LATEST version of each note.
-- HNO_NOTE_TEXT has multiple rows per NOTE_ID -- one per revision
-- (different NOTE_CSN_ID values). MAX(NOTE_CSN_ID) per NOTE_ID is the
-- current/latest revision. Concatenating across versions without this
-- step would duplicate text from earlier drafts.
-- =====================================================================
IF OBJECT_ID('tempdb..#LatestNoteVer') IS NOT NULL DROP TABLE #LatestNoteVer;

SELECT
    t.NOTE_ID
   ,LatestNoteCSN = MAX(t.NOTE_CSN_ID)
INTO #LatestNoteVer
FROM HNO_NOTE_TEXT t
INNER JOIN #NoteIDs n ON n.NOTE_ID = t.NOTE_ID
GROUP BY t.NOTE_ID;

CREATE UNIQUE CLUSTERED INDEX IX_LatestNoteVer ON #LatestNoteVer (NOTE_ID);


-- =====================================================================
-- STEP 4: Reassemble each note's text by concatenating lines in order.
-- =====================================================================
IF OBJECT_ID('tempdb..#NoteAgg') IS NOT NULL DROP TABLE #NoteAgg;

SELECT
    t.NOTE_ID
   ,NoteText = STRING_AGG(CONVERT(NVARCHAR(MAX), t.NOTE_TEXT),
                          CHAR(13) + CHAR(10))
               WITHIN GROUP (ORDER BY t.[Line])
INTO #NoteAgg
FROM HNO_NOTE_TEXT t
INNER JOIN #LatestNoteVer l
    ON  l.NOTE_ID       = t.NOTE_ID
    AND l.LatestNoteCSN = t.NOTE_CSN_ID
GROUP BY t.NOTE_ID;

CREATE UNIQUE CLUSTERED INDEX IX_NoteAgg ON #NoteAgg (NOTE_ID);


-- =====================================================================
-- STEP 5: Final output -- one row per (LogID, note) with metadata and
-- full text content.
-- =====================================================================
SELECT
    n.LogID
   ,n.PAT_ID
   ,n.EncounterCSN
   ,n.SurgeryStart
   ,n.DischargeTime
   ,n.NOTE_ID
   ,n.NoteTypeCode
   ,n.NoteTypeName
   ,n.NoteCreated
   ,n.AuthorID
   ,MinutesAfterSurgeryStart =
        DATEDIFF(MINUTE, n.SurgeryStart, n.NoteCreated)
   ,HoursBeforeDischarge =
        DATEDIFF(MINUTE, n.NoteCreated, n.DischargeTime) / 60.0
   ,a.NoteText
FROM #NoteIDs n
LEFT JOIN #NoteAgg a ON a.NOTE_ID = n.NOTE_ID
ORDER BY n.LogID, n.NoteCreated;


-- =====================================================================
-- Sanity check (optional): note counts per LogID and note type
-- =====================================================================
/*
SELECT
    NoteTypeName   = n.NoteTypeName
   ,n_cases_w_note = COUNT(DISTINCT n.LogID)
   ,n_notes        = COUNT(*)
   ,avg_per_case   = COUNT(*) * 1.0 / NULLIF(COUNT(DISTINCT n.LogID), 0)
   ,kb_avg         = AVG(DATALENGTH(a.NoteText)) / 1024.0
FROM #NoteIDs n
LEFT JOIN #NoteAgg a ON a.NOTE_ID = n.NOTE_ID
GROUP BY n.NoteTypeName
ORDER BY n_notes DESC;
*/
