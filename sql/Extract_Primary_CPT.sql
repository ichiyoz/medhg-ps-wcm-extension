USE Clarity;
/* =====================================================================
   Extract_Primary_CPT.sql
   ---------------------------------------------------------------------
   One row per LogID with its primary (top-RVU, MIS) CPT code.

   PrimaryCPT is ALREADY computed into #CohortLogIDs by
   Cohort_MIS_2020_2026.sql (the row with the highest RVU_WORK in
   CLARITY_TDL_TRAN, restricted to the 47-code MIS list). So this is just
   a projection -- and the cohort export CSV already carries this column.

   Use when you want CPT as a standalone join key for analyses that read
   the A1-A5 / bulk parquets (which do NOT carry it). Run in the same
   Cogito session after the cohort temp table exists.

   Save as logid_primary_cpt.csv (columns: LogID, PrimaryCPT) in the
   DATA_DIR so analysis/*.py can merge it on LogID.
   ===================================================================== */
SELECT LogID, PrimaryCPT
FROM #CohortLogIDs
ORDER BY LogID;
