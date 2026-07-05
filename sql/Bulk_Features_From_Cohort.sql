USE Clarity;
/* =====================================================================
   Bulk_Features_From_Cohort.sql
   ---------------------------------------------------------------------
   Bulk version of PeriOp_NSQIP_Query_CTE_corrected.sql. Instead of
   filtering by a single MRN, it joins to #CohortLogIDs (created by
   Load_CohortTempTable.sql).

   Output: one row per LogID with
       - 40 model features (same schema as PeriOp_NSQIP_model.py's
         fill_values dictionary)
       - ReadmittedWithin30Days (binary label)
       - LogID, PAT_ID, EncounterCSN, SurgeryDate (identifiers)

   Run AFTER Load_CohortTempTable.sql in the SAME Cogito session.
   Save result as bulk_features_with_label.csv.

   Schema alignment: column names match
       PeriOp_NSQIP_model.py:_build_per_case_rows
   so the trained pickle is interchangeable between training and
   serving without further column renames.
   ===================================================================== */


-- ---------------------------------------------------------------------
-- A. Patient + case-level anchors -> MATERIALIZED to an indexed temp table
--    so its OR_LOG/PATIENT/PAT_ENC/HSP joins run ONCE. As a CTE it was
--    re-evaluated by each of the ~15 downstream references (CTEs are not
--    materialized), which was the main driver of the 2h runtime.
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS #OR_Cases;
SELECT
    c.LogID
   ,LOG_ID            = orl.LOG_ID
   ,EncounterCSN      = c.EncounterCSN
   ,c.PAT_ID
   ,PrimaryCPT        = c.PrimaryCPT   -- top-RVU MIS CPT (already in #CohortLogIDs)
   ,PAT_MRN_ID        = pid.IDENTITY_ID
   ,BirthDate         = p.BIRTH_DATE
   ,SEX_C             = p.SEX_C
   ,SurgeryDate       = c.SurgeryDate
   ,orl.PRIMARY_PHYS_ID
   ,ASA_RATE_C        = orl2.ASA_RATE_C
   ,orl.PAT_TYPE_C
   ,HSP_ACCOUNT_ID    = penc.HSP_ACCOUNT_ID
   ,ACCT_BASECLS_HA_C = hsp.ACCT_BASECLS_HA_C
INTO #OR_Cases
FROM #CohortLogIDs c
INNER JOIN OR_LOG orl ON CONVERT(varchar, orl.LOG_ID) = c.LogID
LEFT JOIN OR_LOG_2 orl2 ON orl.LOG_ID = orl2.LOG_ID
INNER JOIN PATIENT p ON orl.PAT_ID = p.PAT_ID
LEFT JOIN PAT_ENC penc ON c.EncounterCSN = penc.PAT_ENC_CSN_ID
LEFT JOIN HSP_ACCOUNT hsp ON penc.HSP_ACCOUNT_ID = hsp.HSP_ACCOUNT_ID
OUTER APPLY (
    SELECT TOP 1 IDENTITY_ID
    FROM IDENTITY_ID
    WHERE PAT_ID = orl.PAT_ID AND IDENTITY_TYPE_ID = 2
) pid;

CREATE CLUSTERED   INDEX IX_ORC_LOG ON #OR_Cases (LOG_ID);
CREATE NONCLUSTERED INDEX IX_ORC_PAT ON #OR_Cases (PAT_ID);
CREATE NONCLUSTERED INDEX IX_ORC_CSN ON #OR_Cases (EncounterCSN);
CREATE NONCLUSTERED INDEX IX_ORC_HSP ON #OR_Cases (HSP_ACCOUNT_ID);

-- ---------------------------------------------------------------------
-- B. In-room time anchor for the 180-day lab window.
-- ---------------------------------------------------------------------
WITH InRoomTimes AS (
    SELECT
        oc.LOG_ID
       ,oc.PAT_ID
       ,InRoomTime = COALESCE(MIN(ct.TRACKING_TIME_IN),
                              MIN(CONVERT(datetime, oc.SurgeryDate)))
    FROM #OR_Cases oc
    LEFT JOIN OR_LOG_CASE_TIMES ct
        ON  ct.LOG_ID         = oc.LOG_ID
        AND ct.TRACKING_EVENT_C = 60     -- 60 = In Room
    GROUP BY oc.LOG_ID, oc.PAT_ID
),

-- ---------------------------------------------------------------------
-- C. Cut-to-close minutes (procedure start -> procedure end).
-- ---------------------------------------------------------------------
CutToClose AS (
    SELECT
        oc.LOG_ID
       ,CutToCloseMin = DATEDIFF(MINUTE,
                                 ps.TRACKING_TIME_IN,
                                 pe.TRACKING_TIME_IN)
    FROM #OR_Cases oc
    LEFT JOIN OR_LOG_CASE_TIMES ps
        ON ps.LOG_ID = oc.LOG_ID AND ps.TRACKING_EVENT_C = 80   -- Proc Start
    LEFT JOIN OR_LOG_CASE_TIMES pe
        ON pe.LOG_ID = oc.LOG_ID AND pe.TRACKING_EVENT_C = 390  -- Proc End
),

-- ---------------------------------------------------------------------
-- D. Other / Concurrent procedure counts (panel-based, same heuristic
--    as the production CTE).
-- ---------------------------------------------------------------------
ProcCounts AS (
    SELECT
        oc.LOG_ID
       ,OtherProcedures      = COUNT(DISTINCT CASE
            WHEN olap.ALL_PROCS_PANEL = 1 AND olap.ORDINAL > 1
            THEN olap.OR_PROC_ID END)
       ,ConcurrentProcedures = COUNT(DISTINCT CASE
            WHEN olap.ALL_PROCS_PANEL > 1
            THEN olap.OR_PROC_ID END)
    FROM #OR_Cases oc
    INNER JOIN OR_LOG_ALL_PROC olap ON olap.LOG_ID = oc.LOG_ID
    GROUP BY oc.LOG_ID
),

-- ---------------------------------------------------------------------
-- E. Encounter height/weight + discharge disposition.
-- ---------------------------------------------------------------------
EncVitals AS (
    SELECT
        oc.LOG_ID
       ,Height_raw = penc.HEIGHT
       ,Weight_raw = penc.WEIGHT
    FROM #OR_Cases oc
    LEFT JOIN PAT_ENC penc ON penc.PAT_ENC_CSN_ID = oc.EncounterCSN
),
DischargeInfo AS (
    SELECT
        oc.LOG_ID
       ,DischargeDisposition = zdd.NAME
    FROM #OR_Cases oc
    LEFT JOIN PAT_ENC_HSP peh ON peh.PAT_ENC_CSN_ID = oc.EncounterCSN
    LEFT JOIN ZC_DISCH_DISP zdd ON zdd.DISCH_DISP_C = peh.DISCH_DISP_C
),

-- ---------------------------------------------------------------------
-- F. Anesthesia case info from DM_ANESTHESIA: AnesType + ASA fallback
--    + the 3 intraop/PACU complications kept as model features.
-- ---------------------------------------------------------------------
DMAnes AS (
    SELECT
        oc.LOG_ID
       ,AnesType            = MAX(anes.AN_TYPE)
       ,ASA_FromAnes        = MAX(anes.ASA_SCORE_C)
       ,CardiacArrestCount  = COALESCE(MAX(anes.CARDIAC_ARREST), 0)
       ,CVAStrokeCount      = COALESCE(MAX(anes.CABG_STROKE), 0)
       ,ReintubationCount   = COALESCE(MAX(anes.REINTUBATION), 0)
    FROM #OR_Cases oc
    LEFT JOIN DM_ANESTHESIA anes
        ON  anes.PAT_ID = oc.PAT_ID
        AND CONVERT(date, anes.RECORD_DATE) = oc.SurgeryDate
    GROUP BY oc.LOG_ID
),

-- ---------------------------------------------------------------------
-- G. Labs: 12 components, most recent within 180 days of in-room time,
--    pivoted to one column per component. Mirrors the production CTE.
-- ---------------------------------------------------------------------
LabBase_raw AS (
    SELECT
        irt.LOG_ID
       ,irt.PAT_ID
       ,lab.COMPONENT_ID
       ,lab.ORD_VALUE   AS ObservationValue
       ,DENSE_RANK() OVER (
            PARTITION BY irt.LOG_ID, lab.COMPONENT_ID
            ORDER BY lab.RESULT_TIME DESC
        ) AS Seq
    FROM InRoomTimes irt
    INNER JOIN ORDER_PROC prc  ON prc.PAT_ID = irt.PAT_ID
    INNER JOIN ORDER_RESULTS lab ON prc.ORDER_PROC_ID = lab.ORDER_PROC_ID
    WHERE lab.COMPONENT_ID IN (
        '17517','123000017517',  -- Albumin
        '45443','123000045443',  -- Hematocrit
        '21600','123000021600',  -- Creatinine
        '30940','123000030940',  -- BUN
        '29512','123000029512',  -- Sodium
        '19208','123000019208',  -- SGOT
        '67686','123000067686',  -- AlkPhos
        '19752','123000019752',  -- Total Bilirubin
        '31732','123000031732',  -- APTT
        '7773','123000007773',   -- Platelets
        '8045','123000008045',   -- WBC
        '63016','123000063016'   -- INR
    )
    AND lab.RESULT_TIME BETWEEN DATEADD(DAY, -180, irt.InRoomTime) AND irt.InRoomTime
),
Labs AS (
    SELECT
        LOG_ID
       ,Albumin    = MAX(CASE WHEN COMPONENT_ID IN ('17517','123000017517')  THEN ObservationValue END)
       ,Hematocrit = MAX(CASE WHEN COMPONENT_ID IN ('45443','123000045443')  THEN ObservationValue END)
       ,Creatinine = MAX(CASE WHEN COMPONENT_ID IN ('21600','123000021600')  THEN ObservationValue END)
       ,BUN_v      = MAX(CASE WHEN COMPONENT_ID IN ('30940','123000030940')  THEN ObservationValue END)
       ,Sodium     = MAX(CASE WHEN COMPONENT_ID IN ('29512','123000029512')  THEN ObservationValue END)
       ,SGOT_v     = MAX(CASE WHEN COMPONENT_ID IN ('19208','123000019208')  THEN ObservationValue END)
       ,AlkPhos    = MAX(CASE WHEN COMPONENT_ID IN ('67686','123000067686')  THEN ObservationValue END)
       ,TotalBili  = MAX(CASE WHEN COMPONENT_ID IN ('19752','123000019752')  THEN ObservationValue END)
       ,APTT_v     = MAX(CASE WHEN COMPONENT_ID IN ('31732','123000031732')  THEN ObservationValue END)
       ,Platelets  = MAX(CASE WHEN COMPONENT_ID IN ('7773','123000007773')   THEN ObservationValue END)
       ,WBC_v      = MAX(CASE WHEN COMPONENT_ID IN ('8045','123000008045')   THEN ObservationValue END)
       ,INR_v      = MAX(CASE WHEN COMPONENT_ID IN ('63016','123000063016')  THEN ObservationValue END)
    FROM LabBase_raw
    WHERE Seq = 1
    GROUP BY LOG_ID
),

-- ---------------------------------------------------------------------
-- H. Comorbidity ICD-10 flags from broadened dx history (problem list +
--    24-mo encounter dx). HF is history-based (no 30-day gate).
-- ---------------------------------------------------------------------
DxHistory AS (
    -- Broadened source: active PROBLEM_LIST UNION encounter/billing
    -- diagnoses (PAT_ENC_DX) within 24 months before surgery. Problem-list-
    -- only under-captured chronic comorbidities (COPD 33%, HF 31%,
    -- disseminated cancer 67% sensitivity vs NSQIP); encounter dx close it.
    -- (a) active problem list as of surgery
    SELECT DISTINCT
        oc.PAT_ID
       ,ICD10_CODE = ed.CODE
    FROM #OR_Cases oc
    INNER JOIN PROBLEM_LIST pl ON pl.PAT_ID = oc.PAT_ID
    INNER JOIN EDG_CURRENT_ICD10 ed ON ed.DX_ID = pl.DX_ID
    WHERE pl.NOTED_DATE <= oc.SurgeryDate
      AND (pl.RESOLVED_DATE IS NULL OR pl.RESOLVED_DATE >= oc.SurgeryDate)
      AND ( ed.CODE LIKE 'E0[89]%' OR ed.CODE LIKE 'E1[013]%'
         OR ed.CODE LIKE 'I1[0-6]%' OR ed.CODE LIKE 'I50%'
         OR ed.CODE LIKE 'J44%'     OR ed.CODE LIKE 'R18%'
         OR ed.CODE LIKE 'C7[789]%' OR ed.CODE LIKE 'D6[5-8]%'
         OR ed.CODE LIKE 'N17%'
         OR ed.CODE IN ('Z99.2','N18.6','Z99.11','Z79.52','Z79.899') )
    UNION
    -- (b) encounter diagnoses within 24 months before surgery
    SELECT DISTINCT
        oc.PAT_ID
       ,ICD10_CODE = ed.CODE
    FROM #OR_Cases oc
    INNER JOIN PAT_ENC penc ON penc.PAT_ID = oc.PAT_ID
        AND penc.CONTACT_DATE >= DATEADD(MONTH, -24, oc.SurgeryDate)
        AND penc.CONTACT_DATE <= oc.SurgeryDate
    INNER JOIN PAT_ENC_DX pdx ON pdx.PAT_ENC_CSN_ID = penc.PAT_ENC_CSN_ID
    INNER JOIN EDG_CURRENT_ICD10 ed ON ed.DX_ID = pdx.DX_ID
    WHERE ed.CODE LIKE 'E0[89]%' OR ed.CODE LIKE 'E1[013]%'
       OR ed.CODE LIKE 'I1[0-6]%' OR ed.CODE LIKE 'I50%'
       OR ed.CODE LIKE 'J44%'     OR ed.CODE LIKE 'R18%'
       OR ed.CODE LIKE 'C7[789]%' OR ed.CODE LIKE 'D6[5-8]%'
       OR ed.CODE LIKE 'N17%'
       OR ed.CODE IN ('Z99.2','N18.6','Z99.11','Z79.52','Z79.899')
),
PLFlags AS (
    SELECT
        PAT_ID
       ,Diabetes_Dx       = MAX(CASE WHEN ICD10_CODE LIKE 'E08%' OR ICD10_CODE LIKE 'E09%'
                                       OR ICD10_CODE LIKE 'E10%' OR ICD10_CODE LIKE 'E11%'
                                       OR ICD10_CODE LIKE 'E13%' THEN 1 ELSE 0 END)
       ,HTN_Dx            = MAX(CASE WHEN ICD10_CODE LIKE 'I10%' OR ICD10_CODE LIKE 'I11%'
                                       OR ICD10_CODE LIKE 'I12%' OR ICD10_CODE LIKE 'I13%'
                                       OR ICD10_CODE LIKE 'I15%' OR ICD10_CODE LIKE 'I16%' THEN 1 ELSE 0 END)
       ,HF_Dx             = MAX(CASE WHEN ICD10_CODE LIKE 'I50%' OR ICD10_CODE = 'I11.0'
                                       OR ICD10_CODE = 'I13.0' OR ICD10_CODE = 'I13.2'
                                    THEN 1 ELSE 0 END)   -- history of HF (option a; no 30d gate)
       ,COPD_Dx           = MAX(CASE WHEN ICD10_CODE LIKE 'J44%' THEN 1 ELSE 0 END)
       ,Ascites_Dx        = MAX(CASE WHEN ICD10_CODE LIKE 'R18%' THEN 1 ELSE 0 END)
       ,DisseminatedCa_Dx = MAX(CASE WHEN ICD10_CODE LIKE 'C77%' OR ICD10_CODE LIKE 'C78%'
                                       OR ICD10_CODE LIKE 'C79%' THEN 1 ELSE 0 END)
       ,Bleeding_Dx       = MAX(CASE WHEN ICD10_CODE LIKE 'D65%' OR ICD10_CODE LIKE 'D66%'
                                       OR ICD10_CODE LIKE 'D67%' OR ICD10_CODE LIKE 'D68%' THEN 1 ELSE 0 END)
       ,AKI_Dx            = MAX(CASE WHEN ICD10_CODE LIKE 'N17%' THEN 1 ELSE 0 END)
       ,Dialysis_Dx       = MAX(CASE WHEN ICD10_CODE = 'Z99.2' OR ICD10_CODE = 'N18.6' THEN 1 ELSE 0 END)
       ,VentDep_Dx        = MAX(CASE WHEN ICD10_CODE = 'Z99.11' THEN 1 ELSE 0 END)
       ,Steroid_Dx        = MAX(CASE WHEN ICD10_CODE = 'Z79.52' OR ICD10_CODE = 'Z79.899' THEN 1 ELSE 0 END)
    FROM DxHistory
    GROUP BY PAT_ID
),

-- ---------------------------------------------------------------------
-- I. Active pre-op medications -> classification flags.
-- ---------------------------------------------------------------------
ActiveMeds AS (
    SELECT
        oc.PAT_ID
       ,UPPER(cm.GENERIC_NAME) AS GENERIC
        -- Chronic = on the med >= 30 days before surgery. Filters out the
        -- one-time perioperative doses (dexamethasone for PONV, prophylactic
        -- heparin) that otherwise inflate the med-based comorbidity flags.
       ,Chronic = CASE WHEN om.START_DATE <= DATEADD(DAY, -30, oc.SurgeryDate)
                       THEN 1 ELSE 0 END
    FROM #OR_Cases oc
    INNER JOIN ORDER_MED om
        ON  om.PAT_ID    = oc.PAT_ID
        AND om.START_DATE <= oc.SurgeryDate
        AND (om.END_DATE     IS NULL OR om.END_DATE     >= oc.SurgeryDate)
        AND (om.DISCON_TIME  IS NULL OR om.DISCON_TIME  >= oc.SurgeryDate)
    LEFT JOIN CLARITY_MEDICATION cm ON om.MEDICATION_ID = cm.MEDICATION_ID
),
MedFlags AS (
    SELECT
        PAT_ID
       ,Insulin_Rx          = MAX(CASE WHEN GENERIC LIKE '%INSULIN%' THEN 1 ELSE 0 END)
       ,Antihypertensive_Rx = MAX(CASE WHEN
              GENERIC LIKE '%LISINOPRIL%' OR GENERIC LIKE '%ENALAPRIL%'
           OR GENERIC LIKE '%LOSARTAN%'   OR GENERIC LIKE '%VALSARTAN%'
           OR GENERIC LIKE '%AMLODIPINE%' OR GENERIC LIKE '%DILTIAZEM%'
           OR GENERIC LIKE '%HYDROCHLOROTHIAZIDE%' OR GENERIC LIKE '%FUROSEMIDE%'
           OR GENERIC LIKE '%METOPROLOL%' OR GENERIC LIKE '%CARVEDILOL%'
           OR GENERIC LIKE '%ATENOLOL%'   OR GENERIC LIKE '%PROPRANOLOL%'
           THEN 1 ELSE 0 END)
       -- Require CHRONIC use and DROP acute perioperative steroids
       -- (dexamethasone/hydrocortisone/methylprednisolone given intraop for
       -- PONV / stress dosing) that flagged ~everyone as immunosuppressed.
       ,Immunosuppressant_Rx = MAX(CASE WHEN Chronic = 1 AND (
              GENERIC LIKE '%PREDNISONE%'    OR GENERIC LIKE '%PREDNISOLONE%'
           OR GENERIC LIKE '%CYCLOSPORINE%'  OR GENERIC LIKE '%TACROLIMUS%'
           OR GENERIC LIKE '%MYCOPHENOLATE%' OR GENERIC LIKE '%AZATHIOPRINE%'
           OR GENERIC LIKE '%METHOTREXATE%')
           THEN 1 ELSE 0 END)
       -- Require CHRONIC use and DROP heparin/enoxaparin (prophylactic DVT
       -- dosing given to nearly every surgical inpatient). Keeps chronic
       -- therapeutic oral anticoagulants = genuine bleeding risk.
       ,Anticoagulant_Rx    = MAX(CASE WHEN Chronic = 1 AND (
              GENERIC LIKE '%WARFARIN%'    OR GENERIC LIKE '%APIXABAN%'
           OR GENERIC LIKE '%RIVAROXABAN%' OR GENERIC LIKE '%DABIGATRAN%'
           OR GENERIC LIKE '%EDOXABAN%')
           THEN 1 ELSE 0 END)
    FROM ActiveMeds
    GROUP BY PAT_ID
),

-- ---------------------------------------------------------------------
-- J. Smoker flag (TOBACCO_USER_C=1 or quit <= 1 year before surgery).
-- ---------------------------------------------------------------------
Smoking AS (
    -- Use the MOST RECENT social-hx before surgery, not MAX over all history.
    -- The old MAX flagged anyone who was EVER a current smoker at any past
    -- visit (a former smoker with an old "current" row got flagged) -> ~2.5x
    -- over-capture. This takes their smoking status as of the latest contact.
    SELECT
        PAT_ID
       ,Smoker_Yes = CASE
            WHEN TOBACCO_USER_C = 1 THEN 1
            WHEN TOBACCO_USER_C = 4
                 AND SMOKING_QUIT_DATE >= DATEADD(YEAR, -1, SurgeryDate) THEN 1
            ELSE 0 END
    FROM (
        SELECT oc.PAT_ID, oc.SurgeryDate, s.TOBACCO_USER_C, s.SMOKING_QUIT_DATE
              ,rn = ROW_NUMBER() OVER (PARTITION BY oc.PAT_ID
                                       ORDER BY s.CONTACT_DATE DESC)
        FROM #OR_Cases oc
        INNER JOIN SOCIAL_HX s
            ON s.PAT_ID = oc.PAT_ID AND s.CONTACT_DATE <= oc.SurgeryDate
    ) x
    WHERE rn = 1
),

-- ---------------------------------------------------------------------
-- K. Pre-op RBC transfusion (within 72h before surgery).
-- ---------------------------------------------------------------------
RBC72h AS (
    -- Transfusions are blood-bank orders (ORDER_PROC + ORD_BLOOD_ADMIN),
    -- NOT medications -- the old ORDER_MED/CLARITY_MEDICATION path matched
    -- nothing (0% sensitivity vs NSQIP). Flag an RBC unit whose transfusion
    -- start (BLOOD_START_INSTANT) falls in the 72h before surgery.
    --
    -- BLOOD_PRODUCT_TYP_C is null in this build, so filter by ISBT-128
    -- BLOOD_PRODUCT_CODE. E0332 / E0336 = Red Blood Cells (the two dominant
    -- codes). VERIFY the remaining E-codes against your blood-bank product
    -- master and ADD any that are RBC (irradiated/leukoreduced/washed) here.
    SELECT
        oc.LOG_ID
       ,RBC_Yes = MAX(CASE
            WHEN oba.BLOOD_START_INSTANT BETWEEN
                     DATEADD(HOUR, -72, irt.InRoomTime) AND irt.InRoomTime
            THEN 1 ELSE 0 END)
    FROM #OR_Cases oc
    INNER JOIN InRoomTimes irt ON irt.LOG_ID = oc.LOG_ID   -- datetime anchor (surgery start)
    INNER JOIN ORDER_PROC op ON op.PAT_ID = oc.PAT_ID
        AND op.ORDER_TIME >= DATEADD(DAY, -7, oc.SurgeryDate)   -- prune ORDER_PROC to the
        AND op.ORDER_TIME <  DATEADD(DAY,  1, oc.SurgeryDate)   -- perioperative window
    INNER JOIN ORD_BLOOD_ADMIN oba
        ON  oba.ORDER_ID    = op.ORDER_PROC_ID
        AND oba.IS_BLOOD_YN = 'Y'
        AND LEFT(oba.BLOOD_PRODUCT_CODE, 5) IN (
                'E0332','E0336'          -- Red Blood Cells; expand after verifying
            )
    GROUP BY oc.LOG_ID
),

-- ---------------------------------------------------------------------
-- L. Index discharge time per encounter (anchor for the 30-day window).
-- ---------------------------------------------------------------------
IndexDischarge AS (
    SELECT
        oc.LOG_ID
       ,oc.PAT_ID
       ,oc.SurgeryDate                       -- carried for NSQIP POD-30 anchoring
       ,DischargeTime = MAX(adt.EFFECTIVE_TIME)
    FROM #OR_Cases oc
    INNER JOIN CLARITY_ADT adt
        ON  adt.PAT_ENC_CSN_ID = oc.EncounterCSN
        AND adt.EVENT_TYPE_C   = 2
        AND adt.EVENT_SUBTYPE_C != 2
    GROUP BY oc.LOG_ID, oc.PAT_ID, oc.SurgeryDate
),

-- ---------------------------------------------------------------------
-- M. 30-day UNPLANNED readmission label.
--
-- Gated to HOSP_ADMSN_TYPE_C IN (3,7,8,13,14) at WCM:
--   3=Emergency, 7=Accident, 8=Urgent, 13=Trauma Center, 14=Trauma
--
-- Excluded as planned: 4=Preoperative, 5=Routine, 9=Same Day Surgery,
-- 12=Elective. Excluded as off-target: 1=Inpatient (catch-all),
-- 2=Outpatient (not an admission), 11=INFO NOT AVAILABLE,
-- 6=L&D, 10=Newborn, 15-17=Psychiatric.
--
-- Aligned to the NSQIP gold standard: unplanned readmission within 30
-- days of the OPERATION (POD 0-30), not 30 days from discharge (CMS HRRP).
-- Validated against manually-abstracted NSQIP: the surgery-date anchor +
-- unplanned gate collapse the all-cause over-calls (see validation notes).
-- Residual disagreement is out-of-system readmits (NSQIP captures; Clarity
-- cannot) and unrelated-to-procedure admits NSQIP judges differently.
-- Expected positive rate ~6-8% on this cohort (vs ~10% all-cause).
-- ---------------------------------------------------------------------
ReadmitLabel AS (
    -- NSQIP-aligned window: an unplanned admission that occurs AFTER the
    -- index discharge (a genuine readmission) AND within 30 days of the
    -- SURGERY date (NSQIP measures POD 0-30 from the operation, not from
    -- discharge). DaysToReadmission is the postoperative day (from surgery)
    -- to match NSQIP. Cases still admitted at POD 30 (LOS > 30) get an
    -- empty window and are correctly 0, as in NSQIP.
    SELECT
        ix.LOG_ID
       ,DaysToReadmission = MIN(
            CASE WHEN peh.HOSP_ADMSN_TYPE_C IN (3, 7, 8, 13, 14)
                 THEN DATEDIFF(DAY, ix.SurgeryDate, adt2.EFFECTIVE_TIME)
                 ELSE NULL
            END
        )
    FROM IndexDischarge ix
    LEFT JOIN CLARITY_ADT adt2
        ON  adt2.PAT_ID         = ix.PAT_ID
        AND adt2.EVENT_TYPE_C   = 1
        AND adt2.EVENT_SUBTYPE_C != 2
        AND adt2.EFFECTIVE_TIME  > ix.DischargeTime               -- after index discharge
        AND adt2.EFFECTIVE_TIME  < DATEADD(DAY, 31, ix.SurgeryDate) -- through end of POD 30 (NSQIP);
                                                                   -- SurgeryDate is midnight, so +31
                                                                   -- exclusive includes all of POD 30
                                                                   -- and stays consistent with the
                                                                   -- DATEDIFF<=30 label criterion
    LEFT JOIN PAT_ENC_HSP peh ON peh.PAT_ENC_CSN_ID = adt2.PAT_ENC_CSN_ID
    GROUP BY ix.LOG_ID
),

-- ---------------------------------------------------------------------
-- SDoH-2: Demographics (race / ethnicity / language / ZIP) keyed PAT_ID.
-- Join pattern mirrors PeriOp_NSQIP model_code.gen_demo_query.
-- ---------------------------------------------------------------------
Demographics AS (
    SELECT
        oc.PAT_ID
       ,Race      = rc.NAME
       ,Ethnicity = eth.NAME
       ,Language  = lang.NAME
       ,ZIP       = pat.ZIP
    FROM (SELECT DISTINCT PAT_ID FROM #OR_Cases) oc
    INNER JOIN PATIENT          pat  ON pat.PAT_ID          = oc.PAT_ID
    LEFT  JOIN PATIENT_RACE     pr1  ON pr1.PAT_ID          = pat.PAT_ID AND pr1.LINE = 1
    LEFT  JOIN ZC_PATIENT_RACE  rc   ON pr1.PATIENT_RACE_C  = rc.PATIENT_RACE_C
    LEFT  JOIN ZC_ETHNIC_GROUP  eth  ON pat.ETHNIC_GROUP_C  = eth.ETHNIC_GROUP_C
    LEFT  JOIN ZC_LANGUAGE      lang ON pat.LANGUAGE_C      = lang.LANGUAGE_C
),

-- ---------------------------------------------------------------------
-- SDoH-3: ICD-10 Z-code flags (Z55-Z65) keyed HSP_ACCOUNT_ID.
-- All dx lines (not just principal). VERIFY REF_BILL_CODE format
-- (dotted 'Z59.0' vs undotted 'Z590') in your build.
-- ---------------------------------------------------------------------
SDoH_Zcodes AS (
    SELECT
        dx.HSP_ACCOUNT_ID
       ,SDOH_Housing_Z   = MAX(CASE WHEN ed.REF_BILL_CODE LIKE 'Z59.0%'
                                       OR ed.REF_BILL_CODE LIKE 'Z59.1%' THEN 1 ELSE 0 END)
       ,SDOH_Food_Z      = MAX(CASE WHEN ed.REF_BILL_CODE LIKE 'Z59.4%' THEN 1 ELSE 0 END)
       ,SDOH_Financial_Z = MAX(CASE WHEN ed.REF_BILL_CODE LIKE 'Z59.5%'
                                       OR ed.REF_BILL_CODE LIKE 'Z59.6%'
                                       OR ed.REF_BILL_CODE LIKE 'Z59.7%' THEN 1 ELSE 0 END)
       ,SDOH_Any_Z       = MAX(CASE WHEN ed.REF_BILL_CODE LIKE 'Z5[5-9]%'
                                       OR ed.REF_BILL_CODE LIKE 'Z6[0-5]%' THEN 1 ELSE 0 END)
    FROM HSP_ACCT_DX_LIST dx
    INNER JOIN (SELECT DISTINCT HSP_ACCOUNT_ID FROM #OR_Cases WHERE HSP_ACCOUNT_ID IS NOT NULL) oc
        ON oc.HSP_ACCOUNT_ID = dx.HSP_ACCOUNT_ID
    LEFT JOIN CLARITY_EDG ed ON dx.DX_ID = ed.DX_ID
    GROUP BY dx.HSP_ACCOUNT_ID
)

-- ---------------------------------------------------------------------
-- SDoH-1 (SCREENING) -- TEMPLATE, not active. Epic SDoH-wheel domains
-- are stored as SmartData Elements / flowsheet rows, with build-specific
-- IDs. Fill in your SDE (SMRTDTA_ELEM_*) or FLO_MEAS_IDs, then add the
-- CTE above, its columns to the SELECT, a join, and config.py.
--
-- Screening AS (
--     SELECT oc.PAT_ID
--          , FoodInsecure   = MAX(CASE WHEN sde.ELEM_VALUE_ID = '<food SDE>'   AND v.SMRTDTA_ELEM_VALUE = 'Yes' THEN 1 ELSE 0 END)
--          , HousingUnstable= MAX(CASE WHEN sde.ELEM_VALUE_ID = '<housing SDE>' AND v.SMRTDTA_ELEM_VALUE = 'Yes' THEN 1 ELSE 0 END)
--          , TransportNeed  = MAX(CASE WHEN sde.ELEM_VALUE_ID = '<transport SDE>' AND v.SMRTDTA_ELEM_VALUE = 'Yes' THEN 1 ELSE 0 END)
--     FROM (SELECT DISTINCT PAT_ID FROM #OR_Cases) oc
--     LEFT JOIN SMRTDTA_ELEM_DATA  sde ON sde.RECORD_ID_NUMERIC = oc.PAT_ID
--     LEFT JOIN SMRTDTA_ELEM_VALUE v   ON v.HLV_ID = sde.HLV_ID
--     GROUP BY oc.PAT_ID
-- ),
-- ---------------------------------------------------------------------

-- ---------------------------------------------------------------------
-- Final assembly: one row per LogID, 40 model features + label + SDoH.
-- Column names exactly match PeriOp_NSQIP_model.py:_build_per_case_rows.
-- ---------------------------------------------------------------------
SELECT
    -- ---- identifiers (not used by the model, kept for join keys) ----
    LogID                                              = oc.LogID
   ,PAT_ID                                             = oc.PAT_ID
   ,EncounterCSN                                       = oc.EncounterCSN
   ,SurgeryDate                                        = oc.SurgeryDate
   ,PAT_MRN_ID                                         = oc.PAT_MRN_ID

    -- ---- 40 model features ----
   ,AgeYears                                           =
        CONVERT(decimal(6,2), DATEDIFF(DAY, oc.BirthDate, oc.SurgeryDate) / 365.25)
   ,Gender                                             =
        CASE WHEN oc.SEX_C = 1 THEN 'F' WHEN oc.SEX_C = 2 THEN 'M' ELSE NULL END
   ,PatientType                                        =
        CASE WHEN oc.ACCT_BASECLS_HA_C = 1 THEN 'I'
             WHEN oc.ACCT_BASECLS_HA_C IN (2, 3) THEN 'O'
             WHEN oc.PAT_TYPE_C IN (1, 3) THEN 'I' ELSE 'O' END
   ,ASAClass                                           =
        CONVERT(varchar(8), COALESCE(dma.ASA_FromAnes, oc.ASA_RATE_C))
   ,AnesType                                           = dma.AnesType
   ,[Discharge Disposition]                            = di.DischargeDisposition
   ,[Height (cm)]                                      =
        -- PAT_ENC.HEIGHT is stored in inches
        ROUND(TRY_CONVERT(decimal(6,2), ev.Height_raw) * 2.54, 1)
   ,[Weight (kg)]                                      =
        -- PAT_ENC.WEIGHT is stored in ounces
        ROUND(TRY_CONVERT(decimal(7,2), ev.Weight_raw) * 0.0283495, 1)
   ,BMI                                                =
        CASE WHEN TRY_CONVERT(decimal(6,2), ev.Height_raw) > 0
             THEN (TRY_CONVERT(decimal(7,2), ev.Weight_raw) * 0.0283495)
                  / POWER((TRY_CONVERT(decimal(6,2), ev.Height_raw) * 2.54) / 100.0, 2)
             ELSE NULL END
   ,CutToClose                                         = ctc.CutToCloseMin
   ,[# of Other Procedures]                            = COALESCE(pc.OtherProcedures, 0)
   ,[# of Concurrent Procedures]                       = COALESCE(pc.ConcurrentProcedures, 0)
   ,[NA]                                               = TRY_CONVERT(decimal(8,2), lab.Sodium)
   ,BUN                                                = TRY_CONVERT(decimal(8,2), lab.BUN_v)
   ,Creat                                              = TRY_CONVERT(decimal(8,2), lab.Creatinine)
   ,ALB                                                = TRY_CONVERT(decimal(8,2), lab.Albumin)
   ,BT                                                 = TRY_CONVERT(decimal(8,2), lab.TotalBili)
   ,SGOT                                               = TRY_CONVERT(decimal(8,2), lab.SGOT_v)
   ,ALKPhos                                            = TRY_CONVERT(decimal(8,2), lab.AlkPhos)
   ,WBC                                                = TRY_CONVERT(decimal(8,2), lab.WBC_v)
   ,HCT                                                = TRY_CONVERT(decimal(8,2), lab.Hematocrit)
   ,PLT                                                = TRY_CONVERT(decimal(8,2), lab.Platelets)
   ,INR                                                = TRY_CONVERT(decimal(8,2), lab.INR_v)
   ,APTT                                               = TRY_CONVERT(decimal(8,2), lab.APTT_v)
   ,[Diabetes Mellitus]                                =
        CASE WHEN pl.Diabetes_Dx = 1 OR md.Insulin_Rx = 1 THEN
                CASE WHEN md.Insulin_Rx = 1 THEN 'Insulin' ELSE 'Non-insulin' END
             ELSE 'No' END
   ,[Current Smoker within 1 year]                     =
        CASE WHEN sm.Smoker_Yes = 1 THEN 'Yes' ELSE 'No' END
   ,[Ventilator Dependent]                             =
        CASE WHEN pl.VentDep_Dx = 1 THEN 'Yes' ELSE 'No' END
   ,[History of Severe COPD]                           =
        CASE WHEN pl.COPD_Dx = 1 THEN 'Yes' ELSE 'No' END
   ,Ascites                                            =
        CASE WHEN pl.Ascites_Dx = 1 THEN 'Yes' ELSE 'No' END
   ,[Heart Failure]                                    =
        CASE WHEN pl.HF_Dx = 1 THEN 'Yes' ELSE 'No' END
   ,[Hypertension requiring medication]                =
        CASE WHEN pl.HTN_Dx = 1 OR md.Antihypertensive_Rx = 1 THEN 'Yes' ELSE 'No' END
   ,[Preop Acute Kidney Injury]                        =
        CASE WHEN pl.AKI_Dx = 1 THEN 'Yes' ELSE 'No' END
   ,[Preop Dialysis]                                   =
        CASE WHEN pl.Dialysis_Dx = 1 THEN 'Yes' ELSE 'No' END
   ,[Disseminated Cancer]                              =
        CASE WHEN pl.DisseminatedCa_Dx = 1 THEN 'Yes' ELSE 'No' END
   ,[Immunosuppressive Therapy]                        =
        CASE WHEN pl.Steroid_Dx = 1 OR md.Immunosuppressant_Rx = 1 THEN 'Yes' ELSE 'No' END
   ,[Bleeding Disorder]                                =
        CASE WHEN pl.Bleeding_Dx = 1 OR md.Anticoagulant_Rx = 1 THEN 'Yes' ELSE 'No' END
   ,[Preop RBC Transfusions (72h)]                     =
        CASE WHEN rbc.RBC_Yes = 1 THEN 'Yes' ELSE 'No' END
   ,[# of Cardiac Arrest Requiring CPR]                = dma.CardiacArrestCount
   ,[# of Stroke/Cerebral Vascular Acccident (CVA)]    = dma.CVAStrokeCount
   ,[# of Postop Unplanned Intubation]                 = dma.ReintubationCount

    -- ---- label ----
   ,DaysToReadmission                                  = rl.DaysToReadmission
   ,ReadmittedWithin30Days                             =
        CASE WHEN rl.DaysToReadmission BETWEEN 1 AND 30 THEN 1 ELSE 0 END

    -- ---- procedure code (model feature; appended last so the existing
    --      column order is unchanged). PrimaryCPT (top-RVU MIS CPT) improves
    --      the tabular model (+~0.005 AUROC, 5/5 CV folds). After re-exporting
    --      with this column, add "PrimaryCPT" to config.BULK_FEATURES_COLUMNS
    --      (-> 48) and to config.MODEL_FEATURE_COLUMNS (-> 39). ----
   ,PrimaryCPT                                         = oc.PrimaryCPT

    -- ---- SDoH features (appended after PrimaryCPT; keep this order in
    --      config.BULK_FEATURES_COLUMNS) ----
   ,Race                                               = dem.Race
   ,Ethnicity                                          = dem.Ethnicity
   ,Language                                           = dem.Language
   ,ZIP                                                = dem.ZIP
   ,SDOH_Housing_Z                                     = COALESCE(zc.SDOH_Housing_Z, 0)
   ,SDOH_Food_Z                                        = COALESCE(zc.SDOH_Food_Z, 0)
   ,SDOH_Financial_Z                                   = COALESCE(zc.SDOH_Financial_Z, 0)
   ,SDOH_Any_Z                                         = COALESCE(zc.SDOH_Any_Z, 0)

FROM #OR_Cases oc
LEFT JOIN CutToClose     ctc ON ctc.LOG_ID = oc.LOG_ID
LEFT JOIN ProcCounts     pc  ON pc.LOG_ID  = oc.LOG_ID
LEFT JOIN EncVitals      ev  ON ev.LOG_ID  = oc.LOG_ID
LEFT JOIN DischargeInfo  di  ON di.LOG_ID  = oc.LOG_ID
LEFT JOIN DMAnes         dma ON dma.LOG_ID = oc.LOG_ID
LEFT JOIN Labs           lab ON lab.LOG_ID = oc.LOG_ID
LEFT JOIN PLFlags        pl  ON pl.PAT_ID  = oc.PAT_ID
LEFT JOIN MedFlags       md  ON md.PAT_ID  = oc.PAT_ID
LEFT JOIN Smoking        sm  ON sm.PAT_ID  = oc.PAT_ID
LEFT JOIN RBC72h         rbc ON rbc.LOG_ID = oc.LOG_ID
LEFT JOIN ReadmitLabel   rl  ON rl.LOG_ID         = oc.LOG_ID
LEFT JOIN Demographics   dem ON dem.PAT_ID        = oc.PAT_ID
LEFT JOIN SDoH_Zcodes    zc  ON zc.HSP_ACCOUNT_ID  = oc.HSP_ACCOUNT_ID
ORDER BY oc.SurgeryDate, oc.LogID;
