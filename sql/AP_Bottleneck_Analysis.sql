/* =====================================================================
   AP_Bottleneck_Analysis.sql
   ---------------------------------------------------------------------
   Anatomic Pathology case-tracking bottleneck queries.
   Maps the Beaker Case Tracking screenshot events to Clarity columns.

   Three query blocks (independent - run any subset):

     1. Case-level turnaround (accession -> received -> signout)
        + breakdown by subspecialty / year.
     2. Task-level intervals (block ordered -> block confirmed -> slide
        ordered -> slide confirmed) from SPEC_TASK_LIST.
     3. Test/order-level TAT (ordered -> result entered -> verified)
        from SPEC_TEST_REL.
   ===================================================================== */


-- =====================================================================
-- BLOCK 1: Case-level turnaround
-- =====================================================================
-- Median + p75 + p95 of:
--   * Accession -> Received       (intake speed)
--   * Received -> Sign-out         (full processing turnaround)
--   * Collected -> Sign-out        (end-to-end from patient encounter)
-- Filtered to AP cases with a sign-out (completed cases only).
-- Stratified by year + subspecialty so you can see drift over time
-- and pinpoint which subspecialties have the longest tail.

WITH CompletedCases AS (
    SELECT
        c.CASE_ID
       ,Year                 = YEAR(c.CASE_ACCESSION_DTTM)
       ,SubspecialtyCode     = c.CASE_SUBSPECIALTY_C
       ,AccessionToReceived  = DATEDIFF(HOUR, c.CASE_ACCESSION_DTTM, c.CASE_RECEIVED_DTTM)
       ,ReceivedToSignout    = DATEDIFF(HOUR, c.CASE_RECEIVED_DTTM,  c.CASE_SIGNOUT_DTTM)
       ,CollectedToSignout   = DATEDIFF(HOUR, c.CASE_COLL_DTTM,      c.CASE_SIGNOUT_DTTM)
       ,AccessionToSignout   = DATEDIFF(HOUR, c.CASE_ACCESSION_DTTM, c.CASE_SIGNOUT_DTTM)
    FROM LAB_CASE_DB_MAIN c
    WHERE c.CASE_SIGNOUT_DTTM   IS NOT NULL
      AND c.CASE_ACCESSION_DTTM IS NOT NULL
      AND c.CASE_ACCESSION_DTTM BETWEEN '2023-01-01' AND '2026-12-31'
)
-- PERCENTILE_CONT in SQL Server is window-only (not a true aggregate),
-- so we compute it OVER (PARTITION BY ...) and de-dupe with DISTINCT.
SELECT DISTINCT
    cc.Year
   ,Subspecialty         = COALESCE(zs.NAME, CONVERT(varchar, cc.SubspecialtyCode), '(none)')
   ,n_cases              = COUNT(*) OVER (PARTITION BY cc.Year, cc.SubspecialtyCode)
   ,med_acc_to_recv_hrs  = PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY AccessionToReceived) OVER (PARTITION BY cc.Year, cc.SubspecialtyCode)
   ,p95_acc_to_recv_hrs  = PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY AccessionToReceived) OVER (PARTITION BY cc.Year, cc.SubspecialtyCode)
   ,med_recv_to_sign_hrs = PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY ReceivedToSignout)   OVER (PARTITION BY cc.Year, cc.SubspecialtyCode)
   ,p95_recv_to_sign_hrs = PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY ReceivedToSignout)   OVER (PARTITION BY cc.Year, cc.SubspecialtyCode)
   ,med_acc_to_sign_hrs  = PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY AccessionToSignout)  OVER (PARTITION BY cc.Year, cc.SubspecialtyCode)
   ,p95_acc_to_sign_hrs  = PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY AccessionToSignout)  OVER (PARTITION BY cc.Year, cc.SubspecialtyCode)
FROM CompletedCases cc
LEFT JOIN ZC_SUBSPECIALTY zs ON zs.INTERNAL_ID = cc.SubspecialtyCode
ORDER BY cc.Year DESC, Subspecialty;


-- =====================================================================
-- BLOCK 2: Task-level intervals  (block ordered -> block confirmed,
--          slide ordered -> slide confirmed, etc.)
-- =====================================================================
-- SPEC_TASK_LIST stores one row per task event. TASK_C identifies the
-- task type (block / slide / stain). TASK_ACTION_C distinguishes
-- ordered vs confirmed vs completed.  You'll want to consult
-- ZC_AP_TASK or similar lookup to map the integer codes - run
--    SELECT TASK_C, TASK_ACTION_C, COUNT(*) FROM SPEC_TASK_LIST
--    GROUP BY TASK_C, TASK_ACTION_C ORDER BY 1, 2;
-- to see which codes are populated, then map them in the CASE below.
--
-- The query below computes:
--   * Per task type (TASK_C), the median + p95 time from TASK_ORD_DTTM
--     (when ordered) to TASK_COMP_UTC_DTTM (when completed).
--   * Number of tasks of each type.

WITH TaskIntervals AS (
    SELECT
        tl.TASK_C
       ,tl.TASK_ACTION_C
       ,OrdToCompMin = DATEDIFF(MINUTE, tl.TASK_ORD_DTTM, tl.TASK_COMP_UTC_DTTM)
    FROM SPEC_TASK_LIST tl
    WHERE tl.TASK_ORD_DTTM      IS NOT NULL
      AND tl.TASK_COMP_UTC_DTTM IS NOT NULL
      AND tl.TASK_DELETED_YN    = 'N'
      AND tl.TASK_ORD_DTTM      BETWEEN '2024-01-01' AND '2026-12-31'
)
SELECT DISTINCT
    TaskName            = COALESCE(zt.NAME,  CONVERT(varchar, ti.TASK_C),        '(none)')
   ,TaskAction          = COALESCE(zta.NAME, CONVERT(varchar, ti.TASK_ACTION_C), '(none)')
   ,n_tasks             = COUNT(*) OVER (PARTITION BY ti.TASK_C, ti.TASK_ACTION_C)
   ,med_ord_to_comp_min = PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY OrdToCompMin) OVER (PARTITION BY ti.TASK_C, ti.TASK_ACTION_C)
   ,p95_ord_to_comp_min = PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY OrdToCompMin) OVER (PARTITION BY ti.TASK_C, ti.TASK_ACTION_C)
FROM TaskIntervals ti
LEFT JOIN ZC_TASK        zt  ON zt.INTERNAL_ID  = ti.TASK_C
LEFT JOIN ZC_TASK_ACTION zta ON zta.INTERNAL_ID = ti.TASK_ACTION_C
ORDER BY n_tasks DESC;


-- =====================================================================
-- BLOCK 3: Test-level TAT (Stain / Slide / Reflex tests)
-- =====================================================================
-- SPEC_TEST_REL is one row per test ordered on a specimen.
--   ORDER_INST_DTTM     -> when the test was ordered
--   LAST_RECV_DTTM      -> when specimen was last received for this test
--   TEST_STATUS_DTTM    -> current status timestamp (often Result Entered)
--   TEST_VER_UTC_DTTM   -> verification time (Final Sign-out)
--   TAT_OVERDUE_DTTM    -> pre-computed TAT threshold
--   REFLEX_TRIGGERED_YN -> was a reflex test ordered from this one?
--
-- Per VERIF_STATUS_C bucket: order -> verify intervals.

WITH ResultedTests AS (
    SELECT
        ORDER_INST_DTTM
       ,TEST_VER_UTC_DTTM
       ,TEST_STATUS_DTTM
       ,LAST_RECV_DTTM
       ,VERIF_STATUS_C
       ,REFLEX_TRIGGERED_YN
       ,WasOverdue = CASE WHEN TEST_VER_UTC_DTTM > TAT_OVERDUE_DTTM THEN 1 ELSE 0 END
       ,OrderToVerifyMin  = DATEDIFF(MINUTE, ORDER_INST_DTTM, TEST_VER_UTC_DTTM)
       ,OrderToStatusMin  = DATEDIFF(MINUTE, ORDER_INST_DTTM, TEST_STATUS_DTTM)
       ,RecvToVerifyMin   = DATEDIFF(MINUTE, LAST_RECV_DTTM,  TEST_VER_UTC_DTTM)
    FROM SPEC_TEST_REL
    WHERE ORDER_INST_DTTM    IS NOT NULL
      AND TEST_VER_UTC_DTTM  IS NOT NULL
      AND ORDER_INST_DTTM    BETWEEN '2024-01-01' AND '2026-12-31'
)
SELECT DISTINCT
    VerifStatus         = COALESCE(zvs.NAME, CONVERT(varchar, rt.VERIF_STATUS_C), '(none)')
   ,REFLEX_TRIGGERED_YN
   ,n_tests             = COUNT(*)              OVER (PARTITION BY rt.VERIF_STATUS_C, rt.REFLEX_TRIGGERED_YN)
   ,pct_overdue         = 100.0 * SUM(WasOverdue) OVER (PARTITION BY rt.VERIF_STATUS_C, rt.REFLEX_TRIGGERED_YN)
                        / NULLIF(COUNT(*)        OVER (PARTITION BY rt.VERIF_STATUS_C, rt.REFLEX_TRIGGERED_YN), 0)
   ,med_ord_to_ver_min  = PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY OrderToVerifyMin) OVER (PARTITION BY rt.VERIF_STATUS_C, rt.REFLEX_TRIGGERED_YN)
   ,p95_ord_to_ver_min  = PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY OrderToVerifyMin) OVER (PARTITION BY rt.VERIF_STATUS_C, rt.REFLEX_TRIGGERED_YN)
   ,med_recv_to_ver_min = PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY RecvToVerifyMin)  OVER (PARTITION BY rt.VERIF_STATUS_C, rt.REFLEX_TRIGGERED_YN)
   ,p95_recv_to_ver_min = PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY RecvToVerifyMin)  OVER (PARTITION BY rt.VERIF_STATUS_C, rt.REFLEX_TRIGGERED_YN)
FROM ResultedTests rt
LEFT JOIN ZC_VERIF_STATUS zvs ON zvs.INTERNAL_ID = rt.VERIF_STATUS_C
ORDER BY VerifStatus, REFLEX_TRIGGERED_YN;


-- =====================================================================
-- BLOCK 4: Per-case full timeline (for visual debugging of one case)
-- =====================================================================
-- Replace @CaseID with a real CASE_ID to dump the full event timeline.
-- Useful for sanity-checking the bottleneck numbers against a known case.

/*
-- Find the CASE_ID for a given CASE_NUM like 'COR-SP25-00311':
--   SELECT REQUISITION_ID AS CASE_ID FROM LAB_CASE_INFO
--   WHERE CASE_NUM = 'COR-SP25-00311';
-- LAB_CASE_DB_MAIN.CASE_ID == LAB_CASE_INFO.REQUISITION_ID at this site.

DECLARE @CaseID numeric(18,0) = 12345;  -- replace

-- Case-level events
SELECT 'Case'         AS Source, EventName = 'Accession',   EventTime = CASE_ACCESSION_DTTM, NULL AS PersonID
  FROM LAB_CASE_DB_MAIN WHERE CASE_ID = @CaseID
UNION ALL SELECT 'Case', 'Received',      CASE_RECEIVED_DTTM, NULL FROM LAB_CASE_DB_MAIN WHERE CASE_ID = @CaseID
UNION ALL SELECT 'Case', 'Collected',     CASE_COLL_DTTM,     NULL FROM LAB_CASE_DB_MAIN WHERE CASE_ID = @CaseID
UNION ALL SELECT 'Case', 'Sign-out',      CASE_SIGNOUT_DTTM,  NULL FROM LAB_CASE_DB_MAIN WHERE CASE_ID = @CaseID

-- Specimen-level events
UNION ALL SELECT 'Specimen', 'Specimen Received (AP)', sm.AP_RECEIVE_UTC_DTTM, sm.AP_RECEIVED_BY_ID
  FROM SPEC_DB_MAIN sm WHERE sm.CASE_ID = @CaseID

-- Task events
UNION ALL SELECT 'Task', 'Task ordered (' + CONVERT(varchar, tl.TASK_C) + ')', tl.TASK_ORD_DTTM, tl.TASK_ORD_USER_ID
  FROM SPEC_TASK_LIST tl
  INNER JOIN SPEC_DB_MAIN sm ON sm.SPECIMEN_ID = tl.SPECIMEN_ID
  WHERE sm.CASE_ID = @CaseID AND tl.TASK_DELETED_YN = 'N'
UNION ALL SELECT 'Task', 'Task completed (' + CONVERT(varchar, tl.TASK_C) + ')', tl.TASK_COMP_UTC_DTTM, tl.TASK_PERSON_ID
  FROM SPEC_TASK_LIST tl
  INNER JOIN SPEC_DB_MAIN sm ON sm.SPECIMEN_ID = tl.SPECIMEN_ID
  WHERE sm.CASE_ID = @CaseID AND tl.TASK_DELETED_YN = 'N' AND tl.TASK_COMP_UTC_DTTM IS NOT NULL

-- Test events
UNION ALL SELECT 'Test', 'Test ordered',  tr.ORDER_INST_DTTM,    NULL
  FROM SPEC_TEST_REL tr INNER JOIN SPEC_DB_MAIN sm ON sm.SPECIMEN_ID = tr.SPECIMEN_ID
  WHERE sm.CASE_ID = @CaseID
UNION ALL SELECT 'Test', 'Test verified', tr.TEST_VER_UTC_DTTM,  NULL
  FROM SPEC_TEST_REL tr INNER JOIN SPEC_DB_MAIN sm ON sm.SPECIMEN_ID = tr.SPECIMEN_ID
  WHERE sm.CASE_ID = @CaseID AND tr.TEST_VER_UTC_DTTM IS NOT NULL

ORDER BY EventTime;
*/
