-- =============================================================================
-- DRAFT -- Comorbidity derivations for the PeriOp_NSQIP model
-- Clarity-native APPROXIMATIONS of NSQIP chart-abstracted comorbidities.
--
-- ***  THIS IS A SCAFFOLD, NOT A VALIDATED ARTIFACT  ***
--   * ICD-10 code sets below are a starting point and MUST be reviewed by your
--     clinical / coding team against the NSQIP variable definitions.
--   * Output values use the model's training vocabulary ('Yes'/'No', and
--     'No'/'Non-insulin'/'Insulin' for diabetes, 'Independent' for functional).
--   * Table/column names (PROBLEM_LIST, EDG_CURRENT_ICD10, SOCIAL_HX, ZC_*) must
--     be confirmed in your Cogito build -- the metadata catalog is unavailable,
--     so these cannot be auto-verified. Comment out any CTE whose tables error.
--   * TRAIN/SERVE CONSISTENCY: the training CSV holds NSQIP-ABSTRACTED values.
--     To use these at serve time without a distribution shift, re-derive the
--     comorbidities the SAME way for the training cohort (replace the CSV's
--     abstracted columns), or accept that derived != abstracted.
--
-- HOW TO USE: paste these CTEs into PeriOp_NSQIP_Query_CTE_corrected.sql after
-- PeriOpBase, add the LEFT JOINs + output columns shown at the bottom.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Active problem-list ICD-10 codes present on/before surgery (preexisting Dx)
-- ---------------------------------------------------------------------------
Comorbid_Dx AS (
    SELECT
        pl.PAT_ID
       ,Diabetes_Dx    = MAX(CASE WHEN icd.CODE LIKE 'E08%' OR icd.CODE LIKE 'E09%'
                                    OR icd.CODE LIKE 'E10%' OR icd.CODE LIKE 'E11%'
                                    OR icd.CODE LIKE 'E13%' THEN 1 ELSE 0 END)
       ,HTN_Dx         = MAX(CASE WHEN icd.CODE LIKE 'I10%' OR icd.CODE LIKE 'I11%'
                                    OR icd.CODE LIKE 'I12%' OR icd.CODE LIKE 'I13%'
                                    OR icd.CODE LIKE 'I15%' OR icd.CODE LIKE 'I16%' THEN 1 ELSE 0 END)
       ,HF_Dx          = MAX(CASE WHEN icd.CODE LIKE 'I50%' OR icd.CODE = 'I11.0'
                                    OR icd.CODE = 'I13.0' OR icd.CODE = 'I13.2' THEN 1 ELSE 0 END)
       ,COPD_Dx        = MAX(CASE WHEN icd.CODE LIKE 'J44%' THEN 1 ELSE 0 END)
       ,Ascites_Dx     = MAX(CASE WHEN icd.CODE LIKE 'R18%' THEN 1 ELSE 0 END)
       ,DisseminatedCa_Dx = MAX(CASE WHEN icd.CODE LIKE 'C77%' OR icd.CODE LIKE 'C78%'
                                    OR icd.CODE LIKE 'C79%' OR icd.CODE LIKE 'C7B%'
                                    OR icd.CODE = 'C80.0' OR icd.CODE = 'C80.1' THEN 1 ELSE 0 END)
       ,Bleeding_Dx    = MAX(CASE WHEN icd.CODE LIKE 'D65%' OR icd.CODE LIKE 'D66%'
                                    OR icd.CODE LIKE 'D67%' OR icd.CODE LIKE 'D68%'
                                    OR icd.CODE LIKE 'D69%' THEN 1 ELSE 0 END)
       ,AKI_Dx         = MAX(CASE WHEN icd.CODE LIKE 'N17%' THEN 1 ELSE 0 END)
       ,Dialysis_Dx    = MAX(CASE WHEN icd.CODE = 'Z99.2' OR icd.CODE = 'N18.6' THEN 1 ELSE 0 END)
       ,VentDep_Dx     = MAX(CASE WHEN icd.CODE = 'Z99.11' THEN 1 ELSE 0 END)
    FROM PROBLEM_LIST pl
    JOIN PeriOpBase peri
        ON pl.PAT_ID = peri.PAT_ID
    JOIN EDG_CURRENT_ICD10 icd
        ON pl.DX_ID = icd.DX_ID
    WHERE pl.PROBLEM_STATUS_C = 1                       -- Active problems only  (VERIFY code)
      AND (pl.NOTED_DATE <= peri.SURGERY_DATE OR pl.NOTED_DATE IS NULL)
    GROUP BY pl.PAT_ID
),

-- ---------------------------------------------------------------------------
-- Current smoker within 1 year (NSQIP). Source: SOCIAL_HX tobacco status.
-- VERIFY: column is often TOBACCO_USER_C or SMOKING_TOB_USE_C; values via
-- ZC_TOBACCO_USER (e.g. 'Yes'/'Current Every Day'/'Current Some Day').
-- ---------------------------------------------------------------------------
Smoking AS (
    SELECT
        sh.PAT_ID
       ,Smoker = MAX(CASE WHEN ztu.NAME LIKE '%Current%' OR ztu.NAME = 'Yes' THEN 1 ELSE 0 END)
    FROM SOCIAL_HX sh
    JOIN PeriOpBase peri
        ON sh.PAT_ID = peri.PAT_ID
    LEFT JOIN ZC_TOBACCO_USER ztu
        ON sh.TOBACCO_USER_C = ztu.TOBACCO_USER_C
    GROUP BY sh.PAT_ID
)

-- ===========================================================================
-- WIRE-IN: add these LEFT JOINs to the final SELECT's FROM block --
--    LEFT JOIN Comorbid_Dx cmb ON p.PAT_ID = cmb.PAT_ID
--    LEFT JOIN Smoking      smk ON p.PAT_ID = smk.PAT_ID
--
-- and these output columns (model training vocabulary) to the SELECT list:
--
--   ,[Diabetes Mellitus]               = CASE WHEN cmb.Diabetes_Dx = 1 THEN 'Non-insulin' ELSE 'No' END  -- TODO insulin split needs meds
--   ,[Hypertension requiring medication]= CASE WHEN cmb.HTN_Dx        = 1 THEN 'Yes' ELSE 'No' END
--   ,[Heart Failure]                   = CASE WHEN cmb.HF_Dx          = 1 THEN 'Yes' ELSE 'No' END
--   ,[History of Severe COPD]          = CASE WHEN cmb.COPD_Dx        = 1 THEN 'Yes' ELSE 'No' END
--   ,[Ascites]                         = CASE WHEN cmb.Ascites_Dx     = 1 THEN 'Yes' ELSE 'No' END
--   ,[Disseminated Cancer]             = CASE WHEN cmb.DisseminatedCa_Dx = 1 THEN 'Yes' ELSE 'No' END
--   ,[Bleeding Disorder]               = CASE WHEN cmb.Bleeding_Dx    = 1 THEN 'Yes' ELSE 'No' END
--   ,[Preop Acute Kidney Injury]       = CASE WHEN cmb.AKI_Dx         = 1 THEN 'Yes' ELSE 'No' END
--   ,[Preop Dialysis]                  = CASE WHEN cmb.Dialysis_Dx    = 1 THEN 'Yes' ELSE 'No' END
--   ,[Ventilator Dependent]            = CASE WHEN cmb.VentDep_Dx     = 1 THEN 'Yes' ELSE 'No' END
--   ,[Current Smoker within 1 year]    = CASE WHEN smk.Smoker         = 1 THEN 'Yes' ELSE 'No' END
--
-- NOT derivable from Dx/social-history alone (need meds / blood bank / flowsheet):
--   * Immunosuppressive Therapy  -> ORDER_MED: steroids (>=10mg prednisone-equiv)
--                                   or immunosuppressants within 30d pre-op
--   * Preop RBC Transfusions(72h)-> blood administration (MAR / blood bank) within 72h
--   * Functional Heath Status    -> nursing ADL assessment flowsheet
--   (leave these dropped, or default 'No'/'Independent' -- but a wrong default
--    biases the model, same issue as Origin Status.)
-- ===========================================================================
