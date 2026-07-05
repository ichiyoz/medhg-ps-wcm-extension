/* =====================================================================
   GNN_Graph_Extract.sql  (v3 - cohort-scoped; all A-queries filter to #CohortLogIDs)
   ---------------------------------------------------------------------
   Heterogeneous graph extract for MedHG-PS-style ie-HGCN training and
   per-MRN runtime embedding lookup.

   Graph (matches Chen et al., npj Health Systems 2025, Fig 6):
       ENC  -- surgical encounter (one per surgery)
       PROV -- provider (surgeon, anesth, CRNA, resident, etc.)
       UNIT -- care unit (acute / intermediate / intensive)

   ---------------------------------------------------------------------
   CHANGES vs v1 (after reviewing Clarity data dictionaries):

   1. OR_LOG_ALL_USERS -> OR_LOG_ALL_SURG. The all-users table is not
      used by your production CTE and is likely not exposed at this
      site. OR_LOG_ALL_SURG is the canonical OR-team-member table; it
      stores all SURGEON-side participants (primary + assistant +
      resident + fellow) keyed by LOG_ID + LINE.

   2. DM_ANES_STAFF -> DM_ANESTHESIA columns. There is no separate
      per-staff anesthesia table at this site. DM_ANESTHESIA itself
      carries:
        RESPONSIBLE_PROV_ID  -- the attending anesthesiologist
        ALL_AN_STAFF         -- semicolon-delimited list of every
                                anesthesia provider on the record
      We split ALL_AN_STAFF on ';' to get one PROV_ID per staff
      member.

   3. ProviderType classification no longer relies on string-matching
      ZC_OR_PANEL_ROLE.NAME. The ROLE_C category code on OR_LOG_ALL_SURG
      ships with only "1 - Primary"; assistant/resident codes are
      site-defined and unreliable. Instead we combine flags from
      CLARITY_SER and CLARITY_SER_2:
        IS_RESIDENT       Y/N
        EMPLOYED_CRNA_YN  Y/N
        HOSPITALIST_YN    Y/N
        DOCTORS_DEGREE    e.g. 'MD', 'DO', 'CRNA'
        MCD_PROF_CD_C     NYSDOH profession code (RN=22, PA=23, ...)
      with explicit precedence rules.

   4. UnitType classification combines Epic-released codes
      (CLARITY_DEP.ADT_UNIT_TYPE_C, INPATIENT_DEPT_YN, OR_UNIT_TYPE_C,
      IS_PERIOP_DEP_YN) with the DEPARTMENT_NAME LIKE fallback. Pre-op
      and post-op (PACU/Phase II) departments are explicitly
      classified as Intermediate per the paper's mapping for PACU.

   5. PROV_ID type was tightened to VARCHAR for the joins.

   ---------------------------------------------------------------------
   STILL UNVERIFIED (no data dictionary uploaded):
     * CLARITY_ADT  -- EVENT_TYPE_C / EVENT_SUBTYPE_C code values.
                       Existing CTE uses IN (1,2) with !=2 for subtypes,
                       so we follow that precedent.
   ===================================================================== */


/* =====================================================================
   PART A -- OFFLINE BULK EXTRACT (training)
   ===================================================================== */

-- PREREQUISITE: run UnitTypeLookup_Temp.sql in THIS tab first to
-- create #UnitTypeLookup before any query below references it.
USE Clarity;   -- required: Cogito tabs may inherit master/tempdb default


-- ---------------------------------------------------------------------
-- A1. ENC (encounter) nodes
--     One row per primary surgical case in the cohort window.
-- ---------------------------------------------------------------------
-- PrimarySurgID from OR_LOG_ALL_SURG ROLE_C=1 (Epic-released: 1=Primary).
-- OR_LOG.PRIMARY_PHYS_ID is mostly NULL at WCM and is NOT the surgeon.
SELECT
    LogID         = CONVERT(varchar, orl.LOG_ID)
   ,EncounterCSN  = poal.PAT_ENC_CSN_ID
   ,PAT_ID        = orl.PAT_ID
   ,SurgeryDate   = CONVERT(date, orl.SURGERY_DATE)
   ,PrimarySurgID = (SELECT TOP 1 CONVERT(varchar, olas.SURG_ID)
                     FROM OR_LOG_ALL_SURG olas
                     WHERE olas.LOG_ID = orl.LOG_ID
                       AND olas.ROLE_C = 1)
FROM OR_LOG orl
INNER JOIN #CohortLogIDs coh
    ON CONVERT(varchar, orl.LOG_ID) = coh.LogID
LEFT JOIN OR_LOG_2 orl2
    ON orl.LOG_ID = orl2.LOG_ID
LEFT JOIN PAT_OR_ADM_LINK poal
    ON orl.LOG_ID = poal.LOG_ID
WHERE orl.STATUS_C = 2
  AND orl2.PROC_NOT_DONE_RSN_C IS NULL
  AND orl.PROC_NOT_PERF_C      IS NULL
;


-- ---------------------------------------------------------------------
-- A2. ENC -- PROV edges  (surgeon + anesthesia OR participants)
--     One row per (LOG_ID, ProvID, RoleSource). The UNION combines:
--       (a) surgical-team rows from OR_LOG_ALL_SURG
--       (b) the attending anesthesiologist from DM_ANESTHESIA.RESPONSIBLE_PROV_ID
--       (c) every other anesthesia staff from the semicolon-delimited
--           DM_ANESTHESIA.ALL_AN_STAFF column
--     ProviderType is derived from CLARITY_SER + CLARITY_SER_2 flags
--     with explicit precedence (see ProviderTypeRule CTE).
-- ---------------------------------------------------------------------
WITH ProviderTypeRule AS (
    -- Reusable per-provider classification. Precedence order:
    --   1. EMPLOYED_CRNA_YN  = 'Y' -> OtherClinician (CRNA)
    --   2. IS_RESIDENT       = 'Y' -> SurgicalTeam   (resident on a surgical role)
    --   3. DOCTORS_DEGREE LIKE 'CRNA' -> OtherClinician
    --   4. MCD_PROF_CD_C IN (22) -> Nurse        -- Registered Professional Nurse
    --      MCD_PROF_CD_C IN (23) -> OtherClinician -- PA
    --      MCD_PROF_CD_C IN (46) -> OtherClinician -- NP Anesthesia
    --      MCD_PROF_CD_C IN (52,53) -> Technician -- Respiratory Tech
    --   5. STAFF_RESOURCE = 'Person' AND DOCTORS_DEGREE LIKE '%MD%' or '%DO%'
    --       -> Physician (surgical context => SurgicalTeam,
    --                     anesthesia context => OtherClinician;
    --       the calling SELECT supplies the context label.)
    --   6. STAFF_RESOURCE = 'Resource' -> Other (rooms, equipment)
    --   7. Default                    -> Other
    SELECT
        ser.PROV_ID
       ,IsCRNA      = CASE WHEN ser.EMPLOYED_CRNA_YN = '1' THEN 1 ELSE 0 END
       ,IsResident  = CASE WHEN ser.IS_RESIDENT     = 'Y' THEN 1 ELSE 0 END
       ,IsPhysician = CASE WHEN ser.DOCTORS_DEGREE LIKE '%MD%'
                            OR ser.DOCTORS_DEGREE LIKE '%DO%'  THEN 1 ELSE 0 END
       ,IsNurseLike = CASE WHEN ser.MCD_PROF_CD_C = 22 THEN 1 ELSE 0 END
       ,IsTechLike  = CASE WHEN ser.MCD_PROF_CD_C IN (52, 53) THEN 1 ELSE 0 END
       ,IsResource  = CASE WHEN ser.STAFF_RESOURCE = '2' THEN 1 ELSE 0 END
       ,DoctorsDegree = ser.DOCTORS_DEGREE
       ,MCDCode       = ser.MCD_PROF_CD_C
    FROM CLARITY_SER ser
)
-- ---- (a) Surgical-team rows from OR_LOG_ALL_SURG ----
SELECT
    LogID         = CONVERT(varchar, olas.LOG_ID)
   ,ProvID        = CONVERT(varchar, olas.SURG_ID)
   ,RoleCode      = olas.ROLE_C                       -- 1 = Primary (Epic-released)
   ,RoleSource    = 'OR_LOG_ALL_SURG'
   ,ProviderType  =
        CASE
            WHEN ptr.IsResident = 1 THEN 'SurgicalTeam'  -- surgical resident
            WHEN ptr.IsCRNA     = 1 THEN 'OtherClinician'
            WHEN ptr.IsPhysician = 1 THEN 'SurgicalTeam' -- attending surgeon
            WHEN ptr.IsNurseLike = 1 THEN 'Nurse'        -- e.g. RN-first-assist
            WHEN ptr.IsTechLike  = 1 THEN 'Technician'
            ELSE 'Other'
        END
FROM OR_LOG_ALL_SURG olas
INNER JOIN #CohortLogIDs coh
    ON CONVERT(varchar, olas.LOG_ID) = coh.LogID
INNER JOIN OR_LOG orl
    ON olas.LOG_ID = orl.LOG_ID
LEFT JOIN ProviderTypeRule ptr
    ON olas.SURG_ID = ptr.PROV_ID
WHERE orl.STATUS_C = 2

UNION ALL

-- ---- (b) Attending anesthesiologist ----
SELECT
    LogID         = CONVERT(varchar, orl.LOG_ID)
   ,ProvID        = CONVERT(varchar, anes.RESPONSIBLE_PROV_ID)
   ,RoleCode      = NULL
   ,RoleSource    = 'DM_ANESTHESIA.RESPONSIBLE_PROV_ID'
   ,ProviderType  =
        CASE
            WHEN ptr.IsCRNA      = 1 THEN 'OtherClinician'
            WHEN ptr.IsResident  = 1 THEN 'OtherClinician'  -- anesth resident
            ELSE 'OtherClinician'                            -- attending anesth
        END
FROM #CohortLogIDs coh
INNER JOIN OR_LOG orl
    ON CONVERT(varchar, orl.LOG_ID) = coh.LogID
INNER JOIN DM_ANESTHESIA anes
    ON  anes.PAT_ID      = orl.PAT_ID
    AND CONVERT(date, anes.RECORD_DATE) = CONVERT(date, orl.SURGERY_DATE)
LEFT JOIN ProviderTypeRule ptr
    ON anes.RESPONSIBLE_PROV_ID = ptr.PROV_ID
WHERE orl.STATUS_C = 2
  AND anes.RESPONSIBLE_PROV_ID IS NOT NULL

UNION ALL

-- ---- (c) Every other anesthesia staff (semicolon-split) ----
-- ALL_AN_STAFF is a VARCHAR(500) of the form "ProvID1;ProvID2;ProvID3".
-- STRING_SPLIT is MS-SQL 2016+. If your Cogito env is older, swap for
-- a CROSS APPLY against a numbers table.
SELECT
    LogID         = CONVERT(varchar, orl.LOG_ID)
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
FROM #CohortLogIDs coh
INNER JOIN OR_LOG orl
    ON CONVERT(varchar, orl.LOG_ID) = coh.LogID
INNER JOIN DM_ANESTHESIA anes
    ON  anes.PAT_ID      = orl.PAT_ID
    AND CONVERT(date, anes.RECORD_DATE) = CONVERT(date, orl.SURGERY_DATE)
CROSS APPLY STRING_SPLIT(anes.ALL_AN_STAFF, ';') split
LEFT JOIN ProviderTypeRule ptr
    ON LTRIM(RTRIM(split.value)) = ptr.PROV_ID
WHERE orl.STATUS_C = 2
  AND anes.ALL_AN_STAFF IS NOT NULL
  AND LEN(LTRIM(RTRIM(split.value))) > 0
;


-- ---------------------------------------------------------------------
-- A3. ENC -- UNIT edges  (full care trajectory: ED arrival → OR → discharge)
--
--     Problem: PAT_OR_ADM_LINK may return the ED encounter CSN for
--     patients admitted via ED; inpatient floor events live under a
--     separate CSN. Joining on CSN alone misses those rows (giving
--     ED-only sequences) or misses the ED (giving post-OR-only).
--
--     Fix: join on PAT_ID + the hospitalization window
--     (HOSP_ADMSN_TIME .. HOSP_DISCHRG_TIME from PAT_ENC on the
--     surgical CSN). This captures ADT events across ALL sub-CSNs
--     within the same hospital stay: ED triage → inpatient admission
--     → OR → PACU → floor → discharge.
--
--     PARTITION BY LOG_ID (not PAT_ENC_CSN_ID) so LEAD/ROW_NUMBER
--     span the full multi-CSN sequence in time order.
-- ---------------------------------------------------------------------
WITH SurgAdmWindow AS (
    -- Anchor each LOG_ID to its hospitalization window.
    -- EncounterCSN comes from #CohortLogIDs (already computed by Cohort SQL).
    -- Fall back to SURGERY_DATE ± 30d if PAT_ENC has no admission/discharge
    -- time (same-day or outpatient cases).
    SELECT
        orl.LOG_ID
       ,orl.PAT_ID
       ,orl.SURGERY_DATE
       ,SurgCSN      = coh.EncounterCSN
       ,AdmTime      = pe.HOSP_ADMSN_TIME
       ,DischTime    = pe.HOSP_DISCHRG_TIME
    FROM #CohortLogIDs coh
    INNER JOIN OR_LOG orl
        ON CONVERT(varchar, orl.LOG_ID) = coh.LogID
    LEFT JOIN OR_LOG_2 orl2
        ON orl.LOG_ID = orl2.LOG_ID
    LEFT JOIN PAT_ENC pe
        ON pe.PAT_ENC_CSN_ID = coh.EncounterCSN
    WHERE orl.STATUS_C = 2
      AND orl2.PROC_NOT_DONE_RSN_C IS NULL
)
-- Include Discharge (EVENT_TYPE_C=2) in the inner CTE so LEAD can
-- anchor the last unit stay's OutTime. Filter it from the final output.
-- Verified pattern per wcm-clarity-adt-events memory.
, RawTrajectory AS (
    SELECT
        saw.LOG_ID
       ,adt.PAT_ENC_CSN_ID
       ,adt.PAT_ID
       ,adt.DEPARTMENT_ID
       ,adt.EVENT_TYPE_C
       ,dep.DEPARTMENT_NAME
       ,COALESCE(ul.GNNUnitType, 'Other')  AS UnitType
       ,ul.UnitType                         AS InstitutionType
       ,adt.EFFECTIVE_TIME                  AS InTime
       ,LEAD(adt.EFFECTIVE_TIME) OVER (
            PARTITION BY saw.LOG_ID
            ORDER BY adt.EFFECTIVE_TIME)    AS OutTime
       ,DATEDIFF(MINUTE, adt.EFFECTIVE_TIME,
                 LEAD(adt.EFFECTIVE_TIME) OVER (
                     PARTITION BY saw.LOG_ID
                     ORDER BY adt.EFFECTIVE_TIME)) / 60.0  AS Hours
       ,ROW_NUMBER() OVER (
            PARTITION BY saw.LOG_ID
            ORDER BY adt.EFFECTIVE_TIME)    AS SeqInEncounter
    FROM SurgAdmWindow saw
    INNER JOIN CLARITY_ADT adt
        ON  adt.PAT_ID = saw.PAT_ID
        AND adt.EFFECTIVE_TIME >= COALESCE(saw.AdmTime,
                                           DATEADD(day, -3, saw.SURGERY_DATE))
        AND adt.EFFECTIVE_TIME <= COALESCE(saw.DischTime,
                                           DATEADD(day, 30, saw.SURGERY_DATE))
    LEFT JOIN CLARITY_DEP dep
        ON adt.DEPARTMENT_ID = dep.DEPARTMENT_ID
    LEFT JOIN #UnitTypeLookup ul
        ON  ul.Clarity_ID = adt.DEPARTMENT_ID
        AND ul.StartDate  <= CAST(adt.EFFECTIVE_TIME AS date)
        AND ul.EndDate    >  CAST(adt.EFFECTIVE_TIME AS date)
    WHERE adt.EVENT_TYPE_C IN (1, 2, 3)  -- 1=Admit, 2=Discharge (LEAD anchor only), 3=Transfer In
      AND adt.EVENT_SUBTYPE_C != 2        -- skip Canceled
)
SELECT
    LogID            = LOG_ID
   ,EncounterCSN     = PAT_ENC_CSN_ID
   ,PAT_ID
   ,DepartmentID     = DEPARTMENT_ID
   ,DepartmentName   = DEPARTMENT_NAME
   ,UnitType
   ,InstitutionType
   ,InTime
   ,OutTime
   ,Hours
   ,SeqInEncounter
FROM RawTrajectory
WHERE EVENT_TYPE_C != 2   -- exclude Discharge rows from output; kept only for LEAD above
;


-- ---------------------------------------------------------------------
-- A4. PROV node attributes
--     Scoped to providers who appear in the cohort via A2 sources:
--     surgical team (OR_LOG_ALL_SURG) or anesthesia (DM_ANESTHESIA).
--     Driving from #CohortLogIDs avoids scanning all of CLARITY_SER
--     and eliminates the slow LIKE '%prov_id%' HAVING filter.
-- ---------------------------------------------------------------------
WITH CohortProvs AS (
    -- Surgical-team providers in the cohort
    SELECT DISTINCT CONVERT(varchar, olas.SURG_ID) AS ProvID
    FROM #CohortLogIDs coh
    INNER JOIN OR_LOG_ALL_SURG olas
        ON CONVERT(varchar, olas.LOG_ID) = coh.LogID

    UNION

    -- Attending anesthesiologist
    SELECT DISTINCT CONVERT(varchar, anes.RESPONSIBLE_PROV_ID)
    FROM #CohortLogIDs coh
    INNER JOIN OR_LOG orl
        ON CONVERT(varchar, orl.LOG_ID) = coh.LogID
    INNER JOIN DM_ANESTHESIA anes
        ON  anes.PAT_ID      = orl.PAT_ID
        AND CONVERT(date, anes.RECORD_DATE) = CONVERT(date, orl.SURGERY_DATE)
    WHERE anes.RESPONSIBLE_PROV_ID IS NOT NULL

    UNION

    -- All anesthesia staff (semicolon-split)
    SELECT DISTINCT LTRIM(RTRIM(split.value))
    FROM #CohortLogIDs coh
    INNER JOIN OR_LOG orl
        ON CONVERT(varchar, orl.LOG_ID) = coh.LogID
    INNER JOIN DM_ANESTHESIA anes
        ON  anes.PAT_ID      = orl.PAT_ID
        AND CONVERT(date, anes.RECORD_DATE) = CONVERT(date, orl.SURGERY_DATE)
    CROSS APPLY STRING_SPLIT(anes.ALL_AN_STAFF, ';') split
    WHERE anes.ALL_AN_STAFF IS NOT NULL
      AND LEN(LTRIM(RTRIM(split.value))) > 0
),
CohortCaseVolume AS (
    -- Case volume across ALL completed MIS surgeries in the cohort window
    -- (not just the sampled cohort) so high-volume surgeons are not
    -- understated due to sampling. 2yr volume uses GETDATE() cutoff.
    SELECT
        CONVERT(varchar, olas.SURG_ID)  AS ProvID
       ,COUNT(DISTINCT CASE WHEN orl.SURGERY_DATE >= DATEADD(YEAR, -2, GETDATE())
                             THEN orl.LOG_ID END)  AS CaseVolume2yr
       ,COUNT(DISTINCT orl.LOG_ID)                 AS CaseVolume5yr
    FROM OR_LOG orl
    INNER JOIN OR_LOG_ALL_SURG olas
        ON olas.LOG_ID = orl.LOG_ID
    WHERE orl.SURGERY_DATE BETWEEN '2022-01-01' AND '2026-12-31'
      AND orl.STATUS_C = 2
    GROUP BY CONVERT(varchar, olas.SURG_ID)
)
SELECT
    ser.PROV_ID         AS ProvID
   ,ser.PROV_NAME       AS ProvName
   ,ser.EMPLOYED_CRNA_YN AS EmployedCRNA
   ,ser.IS_RESIDENT     AS IsResident
   ,ser.HOSPITALIST_YN  AS IsHospitalist
   ,ser.DOCTORS_DEGREE  AS DoctorsDegree
   ,ser.CLINICIAN_TITLE AS ClinicianTitle
   ,ser.PROV_TYPE       AS ProvType
   ,ser.STAFF_RESOURCE  AS StaffResourceCode
   ,ser.MCD_PROF_CD_C   AS MCDProfCode
   ,COALESCE(cv.CaseVolume2yr, 0) AS CaseVolume2yr
   ,COALESCE(cv.CaseVolume5yr, 0) AS CaseVolume5yr
FROM CohortProvs cp
INNER JOIN CLARITY_SER ser
    ON ser.PROV_ID = cp.ProvID
LEFT JOIN CohortCaseVolume cv
    ON cv.ProvID = cp.ProvID
;


-- ---------------------------------------------------------------------
-- A5. UNIT node attributes
-- ---------------------------------------------------------------------
-- UnitType from #UnitTypeLookup (same source as A3) using the most
-- recent active classification per department. COALESCE to 'Other'
-- for any department not in the lookup (estimated ~4% of cohort units).
SELECT
    DepartmentID       = dep.DEPARTMENT_ID
   ,DepartmentName     = dep.DEPARTMENT_NAME
   ,Specialty          = dep.SPECIALTY
   ,InpatientFlag      = dep.INPATIENT_DEPT_YN
   ,ADTUnitTypeCode    = dep.ADT_UNIT_TYPE_C
   ,ORUnitTypeCode     = dep.OR_UNIT_TYPE_C
   ,IsPeriopDept       = dep.IS_PERIOP_DEP_YN
   ,LicensedBeds       = dep.LICENSED_BEDS
   ,ServiceAreaID      = dep.SERV_AREA_ID
   ,UnitType           = COALESCE(ul.GNNUnitType, 'Other')
   ,InstitutionType    = ul.UnitType
FROM CLARITY_DEP dep
LEFT JOIN #UnitTypeLookup ul
    ON  ul.Clarity_ID = dep.DEPARTMENT_ID
    AND ul.EndDate    = '2050-01-01'    -- current active classification only
WHERE dep.DEPARTMENT_ID IN (
        -- Match A3: scope to cohort and use PAT_ID+hospitalization-window join
        -- so ED and cross-CSN departments are captured consistently.
        SELECT DISTINCT adt.DEPARTMENT_ID
        FROM #CohortLogIDs coh
        INNER JOIN OR_LOG orl
            ON CONVERT(varchar, orl.LOG_ID) = coh.LogID
        LEFT JOIN PAT_ENC pe
            ON pe.PAT_ENC_CSN_ID = coh.EncounterCSN
        INNER JOIN CLARITY_ADT adt
            ON  adt.PAT_ID = orl.PAT_ID
            AND adt.EFFECTIVE_TIME >= COALESCE(pe.HOSP_ADMSN_TIME,
                                               DATEADD(day, -3, orl.SURGERY_DATE))
            AND adt.EFFECTIVE_TIME <= COALESCE(pe.HOSP_DISCHRG_TIME,
                                               DATEADD(day, 30, orl.SURGERY_DATE))
        WHERE adt.EVENT_TYPE_C IN (1, 2, 3)
          AND adt.EVENT_SUBTYPE_C != 2
)
;


/* =====================================================================
   PART B -- RUNTIME PER-MRN EXTRACT (inference)
   Three small queries with '{mrn}' placeholder, called from
   PeriOp_NSQIP_model.py at predict time. Returns the providers and
   the unit trajectory for one patient's surgical case(s) so we can
   average pre-trained embeddings.
   ===================================================================== */


-- ---------------------------------------------------------------------
-- B1. Providers for this MRN's surgical case(s)
-- ---------------------------------------------------------------------
WITH PatCases AS (
    SELECT
        orl.LOG_ID
       ,orl.PAT_ID
       ,orl.SURGERY_DATE
    FROM OR_LOG orl
    INNER JOIN PATIENT p
        ON orl.PAT_ID = p.PAT_ID
    LEFT JOIN OR_LOG_2 orl2
        ON orl.LOG_ID = orl2.LOG_ID
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
-- Surgical team
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

-- Attending anesthesiologist
SELECT
    LogID         = CONVERT(varchar, pc.LOG_ID)
   ,ProvID        = CONVERT(varchar, anes.RESPONSIBLE_PROV_ID)
   ,RoleCode      = NULL
   ,RoleSource    = 'DM_ANESTHESIA.RESPONSIBLE_PROV_ID'
   ,ProviderType  = 'OtherClinician'
FROM PatCases pc
INNER JOIN DM_ANESTHESIA anes
    ON  anes.PAT_ID      = pc.PAT_ID
    AND CONVERT(date, anes.RECORD_DATE) = CONVERT(date, pc.SURGERY_DATE)
WHERE anes.RESPONSIBLE_PROV_ID IS NOT NULL

UNION ALL

-- All anesthesia staff (semicolon-split)
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
    ON  anes.PAT_ID      = pc.PAT_ID
    AND CONVERT(date, anes.RECORD_DATE) = CONVERT(date, pc.SURGERY_DATE)
CROSS APPLY STRING_SPLIT(anes.ALL_AN_STAFF, ';') split
LEFT JOIN ProviderTypeRule ptr
    ON LTRIM(RTRIM(split.value)) = ptr.PROV_ID
WHERE anes.ALL_AN_STAFF IS NOT NULL
  AND LEN(LTRIM(RTRIM(split.value))) > 0
;


-- ---------------------------------------------------------------------
-- B2. Care-unit trajectory for this MRN's surgical encounter(s)
-- ---------------------------------------------------------------------
WITH PatCases AS (
    SELECT
        orl.LOG_ID
       ,EncounterCSN = poal.PAT_ENC_CSN_ID
    FROM OR_LOG orl
    INNER JOIN PATIENT p
        ON orl.PAT_ID = p.PAT_ID
    LEFT JOIN PAT_OR_ADM_LINK poal
        ON orl.LOG_ID = poal.LOG_ID
    WHERE p.PAT_MRN_ID = '{mrn}'
      AND poal.PAT_ENC_CSN_ID IS NOT NULL
)
SELECT
    LogID            = CONVERT(varchar, pc.LOG_ID)
   ,EncounterCSN     = adt.PAT_ENC_CSN_ID
   ,DepartmentID     = CONVERT(varchar, adt.DEPARTMENT_ID)
   ,DepartmentName   = dep.DEPARTMENT_NAME
   ,UnitType         =
        CASE
            WHEN dep.ADT_UNIT_TYPE_C = 1                           THEN 'ED'
            WHEN dep.DEPARTMENT_NAME LIKE '%EMERGENCY%'            THEN 'ED'
            WHEN dep.DEPARTMENT_NAME LIKE '%TRIAGE%'               THEN 'ED'
            WHEN dep.OR_UNIT_TYPE_C IN (1, 2)                      THEN 'OR'
            WHEN dep.OR_UNIT_TYPE_C IN (3, 4)                      THEN 'Intermediate'
            WHEN dep.ADT_UNIT_TYPE_C = 7                           THEN 'Intensive'
            WHEN dep.DEPARTMENT_NAME LIKE '%ICU%'                  THEN 'Intensive'
            WHEN dep.DEPARTMENT_NAME LIKE '%CCU%'                  THEN 'Intensive'
            WHEN dep.DEPARTMENT_NAME LIKE '%INTENSIVE%'            THEN 'Intensive'
            WHEN dep.DEPARTMENT_NAME LIKE '%CRITICAL%'             THEN 'Intensive'
            WHEN dep.DEPARTMENT_NAME LIKE '%STEP DOWN%'            THEN 'Intermediate'
            WHEN dep.DEPARTMENT_NAME LIKE '%STEPDOWN%'             THEN 'Intermediate'
            WHEN dep.DEPARTMENT_NAME LIKE '%IMU%'                  THEN 'Intermediate'
            WHEN dep.DEPARTMENT_NAME LIKE '%INTERMEDIATE%'         THEN 'Intermediate'
            WHEN dep.DEPARTMENT_NAME LIKE '%TELEMETRY%'            THEN 'Intermediate'
            WHEN dep.INPATIENT_DEPT_YN = '1'                       THEN 'Acute'
            WHEN dep.ADT_UNIT_TYPE_C = 0                           THEN 'Acute'
            ELSE 'Other'
        END
   ,InTime           = adt.EFFECTIVE_TIME
   ,OutTime          = LEAD(adt.EFFECTIVE_TIME) OVER (
                            PARTITION BY adt.PAT_ENC_CSN_ID
                            ORDER BY adt.EFFECTIVE_TIME)
   ,Hours            = DATEDIFF(MINUTE, adt.EFFECTIVE_TIME,
                                LEAD(adt.EFFECTIVE_TIME) OVER (
                                    PARTITION BY adt.PAT_ENC_CSN_ID
                                    ORDER BY adt.EFFECTIVE_TIME)) / 60.0
   ,SeqInEncounter   = ROW_NUMBER() OVER (
                            PARTITION BY adt.PAT_ENC_CSN_ID
                            ORDER BY adt.EFFECTIVE_TIME)
FROM CLARITY_ADT adt
INNER JOIN PatCases pc
    ON adt.PAT_ENC_CSN_ID = pc.EncounterCSN
LEFT JOIN CLARITY_DEP dep
    ON adt.DEPARTMENT_ID = dep.DEPARTMENT_ID
WHERE adt.EVENT_TYPE_C IN (1, 3)
  AND adt.EVENT_SUBTYPE_C != 2
;


-- ---------------------------------------------------------------------
-- B3. Primary-surgeon volume for cold-start fallback + LLM critique
-- ---------------------------------------------------------------------
WITH PatCases AS (
    SELECT
        orl.LOG_ID
       ,orl.PRIMARY_PHYS_ID
    FROM OR_LOG orl
    INNER JOIN PATIENT p
        ON orl.PAT_ID = p.PAT_ID
    WHERE p.PAT_MRN_ID = '{mrn}'
)
SELECT
    ProvID          = ser.PROV_ID
   ,ProvName        = ser.PROV_NAME
   ,DoctorsDegree   = ser.DOCTORS_DEGREE
   ,IsResident      = ser.IS_RESIDENT
   ,EmployedCRNA    = ser.EMPLOYED_CRNA_YN
   ,IsHospitalist   = ser.HOSPITALIST_YN
   ,CaseVolume2yr   = (
        SELECT COUNT(DISTINCT orl2.LOG_ID)
        FROM OR_LOG orl2
        WHERE orl2.PRIMARY_PHYS_ID = ser.PROV_ID
          AND orl2.SURGERY_DATE >= DATEADD(YEAR, -2, GETDATE())
   )
FROM PatCases pc
INNER JOIN CLARITY_SER ser
    ON pc.PRIMARY_PHYS_ID = ser.PROV_ID
;
