USE Clarity;
/* =====================================================================
   Referral_Patterns_From_Cohort.sql
   ---------------------------------------------------------------------
   Per-case referral fingerprint for the surgical readmission cohort. For
   each case in #CohortLogIDs, captures four proxies of the referral
   pattern that led the patient to the operating surgeon. Intended as a
   case-mix / referral-context feature block, NOT as a provider-
   performance measurement -- results will be interpreted alongside ASA,
   patient type and CPT to disentangle case-mix from routing.

   Output: one row per LogID with
       - LogID, PAT_ID, EncounterCSN, PrimarySurgID, SurgeryDate
       -- Q1) Referring provider identity + specialty
       - ReferringProvID, ReferringProvSpecialty, ReferralSource
             (Internal / External / Self / Unknown)
       -- Q2) Depth of surgical workup
       - n_preop_surg_clinic_visits_90d
       - days_first_surg_clinic_to_surgery
       -- Q3) Within-service handoff (triage from generalist to
       --     subspecialist within the surgical service)
       - FirstSurgeryConsultProvID
       - within_service_handoff        (1 if the surgeon at first
                                        surgical clinic visit differs
                                        from the operating surgeon)
       -- Q4) Formal Epic REFERRAL record linking the preop chain
       - HasEpicReferralRecord         (1 if any REFERRAL row exists
                                        for the patient with a preop
                                        appointment linked to the
                                        surgical service)
       - EpicReferralTypeCode

   Anchor is the preop surgical clinic visit CLOSEST TO (but no later
   than) OR_LOG.SURGERY_DATE. Uses a 90-day lookback for workup depth.

   Run AFTER Load_CohortTempTable.sql in the SAME Cogito session.
   Save result as referral_patterns.csv (one row per LogID).

   Data dictionary references:
     * PAT_ENC.pdf         REFERRING_PROV_ID, DEPARTMENT_ID
     * CLARITY_SER.pdf     PROV_ID, PROV_TYPE, external SPECIALTY resolved
                           via CLARITY_SER_2.PRIMARY_DEPT_ID -> CLARITY_DEP.SPECIALTY
                           (see [[wcm-clarity-schema]] memory)
     * REFERRAL            REFERRAL_TYPE_C, REFERRAL_SPEC_ID, REF_APPT_ID
   ===================================================================== */


-- ---------------------------------------------------------------------
-- A. Case-level anchors (LogID -> operating surgeon, surgery date, CSN).
-- ---------------------------------------------------------------------
IF OBJECT_ID('tempdb..#CaseAnchor') IS NOT NULL DROP TABLE #CaseAnchor;
SELECT  ol.LOG_ID                                                     AS LogID
       ,ol.PAT_ID                                                     AS PAT_ID
       ,COALESCE(ol.OR_LINK_CSN, ol.PAT_ENC_CSN_ID)                   AS EncounterCSN
       ,ol.PRIMARY_PHYS_ID                                            AS PrimarySurgID
       ,CAST(ol.SURGERY_DATE AS date)                                 AS SurgeryDate
INTO    #CaseAnchor
FROM    OR_LOG ol
INNER JOIN #CohortLogIDs c
    ON  c.LogID = ol.LOG_ID;
CREATE CLUSTERED INDEX ix_ca_log ON #CaseAnchor (LogID);
CREATE INDEX           ix_ca_pat ON #CaseAnchor (PAT_ID, SurgeryDate);


-- ---------------------------------------------------------------------
-- B. Surgical-service departments (used to identify preop clinic visits).
--    Anything whose CLARITY_DEP.SPECIALTY starts with 'SURG' or matches
--    curated surgical-service specialties. Materialized so the
--    downstream self-join is cheap.
-- ---------------------------------------------------------------------
IF OBJECT_ID('tempdb..#SurgDept') IS NOT NULL DROP TABLE #SurgDept;
SELECT  dep.DEPARTMENT_ID
       ,dep.SPECIALTY
INTO    #SurgDept
FROM    CLARITY_DEP dep
WHERE   dep.SPECIALTY IS NOT NULL
  AND   (dep.SPECIALTY LIKE 'SURG%'
      OR dep.SPECIALTY LIKE '%SURGERY%'
      OR dep.SPECIALTY IN
             ('General Surgery','Colorectal Surgery','Vascular Surgery'
             ,'Cardiothoracic Surgery','Plastic Surgery','Bariatric Surgery'
             ,'Surgical Oncology','Trauma Surgery','Transplant Surgery'
             ,'Endocrine Surgery','Breast Surgery','Otolaryngology'
             ,'Urology','Orthopedic Surgery','Neurosurgery','Gynecology'));
CREATE CLUSTERED INDEX ix_sd_dep ON #SurgDept (DEPARTMENT_ID);


-- ---------------------------------------------------------------------
-- C. Preop surgical clinic visits: any PAT_ENC in a surgical department,
--    for the same patient, within 90 days BEFORE surgery. Provides both
--    (i) workup depth and (ii) the first-consulting surgeon.
-- ---------------------------------------------------------------------
IF OBJECT_ID('tempdb..#PreopSurgVisits') IS NOT NULL DROP TABLE #PreopSurgVisits;
SELECT  ca.LogID
       ,pe.PAT_ENC_CSN_ID                                             AS ClinicCSN
       ,CAST(pe.CONTACT_DATE AS date)                                 AS VisitDate
       ,pe.VISIT_PROV_ID                                              AS VisitProvID
       ,pe.REFERRING_PROV_ID                                          AS ReferringProvID
       ,pe.DEPARTMENT_ID
       ,sd.SPECIALTY                                                  AS VisitSpecialty
       ,ROW_NUMBER() OVER (PARTITION BY ca.LogID
                           ORDER BY pe.CONTACT_DATE ASC, pe.PAT_ENC_CSN_ID ASC) AS visit_order
INTO    #PreopSurgVisits
FROM    #CaseAnchor ca
INNER JOIN PAT_ENC pe
    ON  pe.PAT_ID = ca.PAT_ID
    AND pe.CONTACT_DATE >= DATEADD(DAY, -90, ca.SurgeryDate)
    AND pe.CONTACT_DATE <  ca.SurgeryDate
INNER JOIN #SurgDept sd
    ON  sd.DEPARTMENT_ID = pe.DEPARTMENT_ID;
CREATE CLUSTERED INDEX ix_psv_log ON #PreopSurgVisits (LogID, visit_order);


-- ---------------------------------------------------------------------
-- D. Anchor visit -- the surgical clinic visit CLOSEST TO surgery
--    (visit_order last for that LogID). Provides the anchor
--    ReferringProvID for the case.
-- ---------------------------------------------------------------------
IF OBJECT_ID('tempdb..#AnchorVisit') IS NOT NULL DROP TABLE #AnchorVisit;
SELECT  v.LogID
       ,v.ClinicCSN                                                   AS AnchorClinicCSN
       ,v.VisitDate                                                   AS AnchorVisitDate
       ,v.VisitProvID                                                 AS AnchorSurgConsultProvID
       ,v.ReferringProvID                                             AS ReferringProvID
INTO    #AnchorVisit
FROM    #PreopSurgVisits v
INNER JOIN (
    SELECT  LogID, MAX(visit_order) AS max_order
    FROM    #PreopSurgVisits
    GROUP BY LogID
) mx ON mx.LogID = v.LogID AND mx.max_order = v.visit_order;


-- ---------------------------------------------------------------------
-- E. First surgical consult (visit_order = 1) -- used for the
--    within-service handoff flag.
-- ---------------------------------------------------------------------
IF OBJECT_ID('tempdb..#FirstConsult') IS NOT NULL DROP TABLE #FirstConsult;
SELECT  LogID
       ,VisitProvID                                                   AS FirstSurgeryConsultProvID
       ,VisitDate                                                     AS FirstSurgConsultDate
FROM    #PreopSurgVisits
WHERE   visit_order = 1;


-- ---------------------------------------------------------------------
-- F. Referring-provider specialty. CLARITY_SER.SPECIALTY_C does NOT
--    exist at WCM (see [[wcm-clarity-schema]] memory); reach specialty
--    via CLARITY_SER_2.PRIMARY_DEPT_ID -> CLARITY_DEP.SPECIALTY.
-- ---------------------------------------------------------------------
IF OBJECT_ID('tempdb..#RefProvSpec') IS NOT NULL DROP TABLE #RefProvSpec;
SELECT  ser.PROV_ID
       ,ser.PROV_TYPE
       ,ser.ACTIVE_STATUS
       ,dep.SPECIALTY                                                 AS RefProvSpecialty
INTO    #RefProvSpec
FROM    CLARITY_SER ser
LEFT JOIN CLARITY_SER_2 ser2 ON ser2.PROV_ID       = ser.PROV_ID
LEFT JOIN CLARITY_DEP   dep  ON dep.DEPARTMENT_ID  = ser2.PRIMARY_DEPT_ID;
CREATE CLUSTERED INDEX ix_rps_prov ON #RefProvSpec (PROV_ID);


-- ---------------------------------------------------------------------
-- G. Formal Epic REFERRAL record within 90d preop that touched the
--    surgical service. REF_APPT_ID typically joins to PAT_ENC_APPT.
--    NOTE: REFERRAL table populates for internal Epic-order-based
--    referrals -- misses external / self / walk-in referrals.
-- ---------------------------------------------------------------------
IF OBJECT_ID('tempdb..#EpicRefFlag') IS NOT NULL DROP TABLE #EpicRefFlag;
SELECT  ca.LogID
       ,MAX(CASE WHEN r.REFERRAL_ID IS NOT NULL THEN 1 ELSE 0 END)    AS HasEpicReferralRecord
       ,MAX(r.REFERRAL_TYPE_C)                                        AS EpicReferralTypeCode
INTO    #EpicRefFlag
FROM    #CaseAnchor ca
LEFT JOIN REFERRAL r
    ON  r.PAT_ID = ca.PAT_ID
    AND r.ENTRY_DATE >= DATEADD(DAY, -90, ca.SurgeryDate)
    AND r.ENTRY_DATE <  ca.SurgeryDate
LEFT JOIN #SurgDept sd
    ON  sd.DEPARTMENT_ID = r.REFERRED_TO_DEPT_ID
GROUP BY ca.LogID;


-- ---------------------------------------------------------------------
-- H. Workup-depth aggregate.
-- ---------------------------------------------------------------------
IF OBJECT_ID('tempdb..#WorkupDepth') IS NOT NULL DROP TABLE #WorkupDepth;
SELECT  v.LogID
       ,COUNT(DISTINCT v.ClinicCSN)                                   AS n_preop_surg_clinic_visits_90d
       ,DATEDIFF(DAY, MIN(v.VisitDate), MAX(ca.SurgeryDate))          AS days_first_surg_clinic_to_surgery
INTO    #WorkupDepth
FROM    #PreopSurgVisits v
INNER JOIN #CaseAnchor  ca ON ca.LogID = v.LogID
GROUP BY v.LogID;


-- ---------------------------------------------------------------------
-- I. Final SELECT.
-- ---------------------------------------------------------------------
SELECT  ca.LogID
       ,ca.PAT_ID
       ,ca.EncounterCSN
       ,ca.PrimarySurgID
       ,ca.SurgeryDate
       -- Q1: referring provider identity + specialty
       ,av.ReferringProvID
       ,rps.RefProvSpecialty                                          AS ReferringProvSpecialty
       ,rps.PROV_TYPE                                                 AS ReferringProvType
       ,CASE
            WHEN av.ReferringProvID IS NULL                     THEN 'Unknown'
            WHEN av.ReferringProvID = ca.PAT_ID                 THEN 'Self'
            WHEN rps.PROV_ID IS NOT NULL                        THEN 'Internal'
            ELSE                                                     'External'
        END                                                           AS ReferralSource
       -- Q2: depth of surgical workup
       ,COALESCE(wd.n_preop_surg_clinic_visits_90d, 0)                AS n_preop_surg_clinic_visits_90d
       ,wd.days_first_surg_clinic_to_surgery
       -- Q3: within-service handoff
       ,fc.FirstSurgeryConsultProvID
       ,CASE
            WHEN fc.FirstSurgeryConsultProvID IS NULL             THEN NULL
            WHEN fc.FirstSurgeryConsultProvID = ca.PrimarySurgID  THEN 0
            ELSE                                                       1
        END                                                           AS within_service_handoff
       -- Q4: formal Epic referral
       ,COALESCE(erf.HasEpicReferralRecord, 0)                        AS HasEpicReferralRecord
       ,erf.EpicReferralTypeCode
FROM    #CaseAnchor        ca
LEFT JOIN #AnchorVisit     av  ON av.LogID = ca.LogID
LEFT JOIN #FirstConsult    fc  ON fc.LogID = ca.LogID
LEFT JOIN #WorkupDepth     wd  ON wd.LogID = ca.LogID
LEFT JOIN #EpicRefFlag     erf ON erf.LogID = ca.LogID
LEFT JOIN #RefProvSpec     rps ON rps.PROV_ID = av.ReferringProvID
ORDER BY ca.LogID;

-- Cleanup
DROP TABLE IF EXISTS #CaseAnchor;
DROP TABLE IF EXISTS #SurgDept;
DROP TABLE IF EXISTS #PreopSurgVisits;
DROP TABLE IF EXISTS #AnchorVisit;
DROP TABLE IF EXISTS #FirstConsult;
DROP TABLE IF EXISTS #WorkupDepth;
DROP TABLE IF EXISTS #EpicRefFlag;
DROP TABLE IF EXISTS #RefProvSpec;
