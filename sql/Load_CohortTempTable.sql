USE Clarity;
/* =====================================================================
   Load_CohortTempTable.sql
   ---------------------------------------------------------------------
   Rebuilds the #CohortLogIDs temp table from the saved cohort CSV
   (cohort_mis_2020_2026.csv, produced by Cohort_MIS_2020_2026.sql).

   WHY THIS EXISTS
   ---------------
   Cohort_MIS_2020_2026.sql draws a STRATIFIED RANDOM sample via NEWID().
   Re-running it produces a DIFFERENT cohort every time. To reproduce the
   exact cohort that was exported and used for training/labeling, you must
   reload the saved CSV -- not re-run the sampler.

   WHEN TO RUN
   -----------
   First, in any new Cogito session, BEFORE these scripts:
       - Bulk_Features_From_Cohort.sql
       - Extract_Primary_CPT.sql
       - GNN_Graph_Extract.sql
       - OR_Notes_Before_Discharge.sql
   They all join to #CohortLogIDs and assume it already exists.

   Temp tables are session-scoped, so re-run this whenever you reconnect.

   SCHEMA (matches the SELECT ... INTO #CohortLogIDs in
   Cohort_MIS_2020_2026.sql; LogID is varchar because every downstream
   join does CONVERT(varchar, orl.LOG_ID) = c.LogID):
       LogID         varchar(30)
       EncounterCSN  numeric(18,0)
       PAT_ID        varchar(18)
       SurgeryDate   date
       SurgeryYear   int
       PrimaryCPT    varchar(10)
       AgeYears      decimal(5,2)
   ===================================================================== */

IF OBJECT_ID('tempdb..#CohortLogIDs') IS NOT NULL DROP TABLE #CohortLogIDs;

CREATE TABLE #CohortLogIDs (
    LogID         varchar(30)    NOT NULL,
    EncounterCSN  numeric(18, 0) NULL,
    PAT_ID        varchar(18)    NULL,
    SurgeryDate   date           NULL,
    SurgeryYear   int            NULL,
    PrimaryCPT    varchar(10)    NULL,
    AgeYears      decimal(5, 2)  NULL
);

-- =====================================================================
-- LOAD PATH A (default): BULK INSERT from the exported CSV.
-- ---------------------------------------------------------------------
-- Requires:
--   * The CSV reachable from the SQL SERVER (UNC share or a path the
--     server's service account can read -- NOT your local C: drive
--     unless the server is local). If Clarity is remote, this WILL fail
--     with "Operating system error ... cannot find the path"; use
--     Path C below instead.
--   * ADMINISTER BULK OPERATIONS permission (or the bulkadmin role).
--
-- Configured for the actual export:
--   Cohort2020-2026.csv -- NO header (FIRSTROW = 1), UTF-8 with BOM
--   (CODEPAGE '65001' strips the BOM so it doesn't corrupt LogID on
--   row 1), 7 columns in schema order, CRLF row endings.
-- =====================================================================
BULK INSERT #CohortLogIDs
FROM 'C:\Users\yiz2014\Downloads\medhg_ps_data\Cohort2020-2026.csv'   -- <-- edit if server-side path differs
WITH (
    FORMAT          = 'CSV',
    CODEPAGE        = '65001',   -- UTF-8; skips the leading BOM
    FIRSTROW        = 1,         -- no header row in this export
    FIELDTERMINATOR = ',',
    ROWTERMINATOR   = '0x0d0a',  -- CRLF (Windows export)
    TABLOCK,
    MAXERRORS       = 0
);

/* ---------------------------------------------------------------------
   LOAD PATH B (fallback): OPENROWSET, if BULK INSERT path/permission
   is blocked but the file is server-reachable. Requires the
   Microsoft.ACE.OLEDB.12.0 provider OR the BULK rowset provider.

INSERT INTO #CohortLogIDs (LogID, EncounterCSN, PAT_ID, SurgeryDate,
                           SurgeryYear, PrimaryCPT, AgeYears)
SELECT LogID, EncounterCSN, PAT_ID, SurgeryDate, SurgeryYear,
       PrimaryCPT, AgeYears
FROM OPENROWSET(
    BULK 'C:\Users\yiz2014\Downloads\medhg_ps_data\Cohort2020-2026.csv',
    FORMAT = 'CSV',
    CODEPAGE = '65001',
    FIRSTROW = 1
) AS src (LogID, EncounterCSN, PAT_ID, SurgeryDate, SurgeryYear,
          PrimaryCPT, AgeYears);
--------------------------------------------------------------------- */

/* ---------------------------------------------------------------------
   LOAD PATH C (no server file access at all): import the CSV with the
   SSMS Import Wizard (or a client tool) into a PERMANENT staging table
   first -- e.g. dbo.CohortStaging in a scratch DB you can write to --
   then run:

INSERT INTO #CohortLogIDs (LogID, EncounterCSN, PAT_ID, SurgeryDate,
                           SurgeryYear, PrimaryCPT, AgeYears)
SELECT LogID, EncounterCSN, PAT_ID, SurgeryDate, SurgeryYear,
       PrimaryCPT, AgeYears
FROM <scratch_db>.dbo.CohortStaging;
--------------------------------------------------------------------- */

-- Same indexes Cohort_MIS_2020_2026.sql creates, so downstream joins
-- on LogID and EncounterCSN stay fast.
CREATE NONCLUSTERED INDEX IX_CohortLogIDs_LogID ON #CohortLogIDs (LogID);
CREATE NONCLUSTERED INDEX IX_CohortLogIDs_CSN   ON #CohortLogIDs (EncounterCSN);

-- Sanity check: row count + year breakdown should match the export.
SELECT TotalRows = COUNT(*) FROM #CohortLogIDs;
SELECT SurgeryYear, n = COUNT(*)
FROM #CohortLogIDs
GROUP BY SurgeryYear
ORDER BY SurgeryYear;
