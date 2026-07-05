USE Clarity;
/* =====================================================================
   Order_Sequence_Extract.sql
   ---------------------------------------------------------------------
   Per-case time-ordered sequence of ORDERS (meds + procedures) as the
   GNN/GRU substrate -- an alternative to the care-unit sequence (A3).

   Design (per the design choices):
     * Source : ORDER_MED (medications) + ORDER_PROC (labs/imaging/proc/
                nursing), interleaved into one time-ordered stream.
     * Token  : grouped class, NOT raw order -> compact vocabulary.
                MED  -> therapeutic class name  ('MED:<class>')
                PROC -> order type name         ('PROC:<type>')
     * Window : ED arrival -> discharge if the encounter came through the
                ED, else admit -> discharge. Anchored on the encounter's
                first ADT event (= ED arrival or admit) and its discharge
                event -- no ED-specific columns required.

   Orders are pulled by PAT_ID + time window (not by CSN) so orders placed
   under an ED CSN distinct from the inpatient CSN are still captured.

   PREREQUISITES (same session): #CohortLogIDs (Load_CohortTempTable.sql).
   Output: one row per (LogID, order); save headerless as
   order_sequence.csv.

   VERIFY in your build (flagged inline): the med therapeutic-class join
   and the ORDER_MED order-time column.
   ===================================================================== */

-- =====================================================================
-- STEP 1: Episode window per case.
--   EpiStart = first ADT event for the encounter (ED arrival if admitted
--              through ED on this CSN, otherwise the admission).
--   EpiEnd   = last discharge ADT event.
-- Cases with no ADT discharge (pure ambulatory) drop out -- see note.
-- =====================================================================
DROP TABLE IF EXISTS #OrderScope;
SELECT
    cl.LogID,
    cl.PAT_ID,
    cl.EncounterCSN,
    EpiStart = adt.FirstEvent,
    EpiEnd   = adt.DischargeTime
INTO #OrderScope
FROM #CohortLogIDs cl
CROSS APPLY (
    SELECT
        FirstEvent    = MIN(a.EFFECTIVE_TIME),
        DischargeTime = MAX(CASE WHEN a.EVENT_TYPE_C = 2 THEN a.EFFECTIVE_TIME END)
    FROM CLARITY_ADT a WITH (NOLOCK)
    WHERE a.PAT_ENC_CSN_ID = cl.EncounterCSN
      AND (a.EVENT_SUBTYPE_C IS NULL OR a.EVENT_SUBTYPE_C != 2)   -- exclude canceled
) adt
WHERE adt.DischargeTime IS NOT NULL
  AND adt.FirstEvent   IS NOT NULL;

CREATE CLUSTERED INDEX IX_OrderScope_PAT ON #OrderScope (PAT_ID);


-- =====================================================================
-- STEP 2: Orders in the window (by PAT_ID + time -> captures ED + IP).
-- =====================================================================
DROP TABLE IF EXISTS #Orders;

-- (a) Procedures / labs / imaging / nursing -- grouped by order type.
SELECT
    s.LogID,
    s.PAT_ID,
    s.EncounterCSN,
    OrderSource = CAST('PROC' AS varchar(4)),
    OrderTime   = op.ORDER_TIME,
    OrderGroup  = 'PROC:' + COALESCE(ot.NAME, 'Other')
INTO #Orders
FROM #OrderScope s
INNER JOIN ORDER_PROC op WITH (NOLOCK)
    ON  op.PAT_ID     = s.PAT_ID
    AND op.ORDER_TIME >= s.EpiStart
    AND op.ORDER_TIME <= s.EpiEnd
LEFT JOIN ZC_ORDER_TYPE ot WITH (NOLOCK)     -- VERIFY: ZC_ORDER_TYPE.ORDER_TYPE_C / NAME
    ON ot.ORDER_TYPE_C = op.ORDER_TYPE_C
WHERE op.ORDER_TIME IS NOT NULL

UNION ALL

-- (b) Medications -- grouped by therapeutic class.
SELECT
    s.LogID,
    s.PAT_ID,
    s.EncounterCSN,
    OrderSource = CAST('MED' AS varchar(4)),
    OrderTime   = om.ORDER_INST,             -- VERIFY: ORDER_MED.ORDER_INST (else ORDERING_DATE)
    OrderGroup  = 'MED:' + COALESCE(tc.NAME, 'Other')
FROM #OrderScope s
INNER JOIN ORDER_MED om WITH (NOLOCK)
    ON  om.PAT_ID     = s.PAT_ID
    AND om.ORDER_INST >= s.EpiStart
    AND om.ORDER_INST <= s.EpiEnd
LEFT JOIN CLARITY_MEDICATION cm WITH (NOLOCK)
    ON cm.MEDICATION_ID = om.MEDICATION_ID
LEFT JOIN ZC_THERA_CLASS tc WITH (NOLOCK)
    ON tc.THERA_CLASS_C = cm.THERA_CLASS_C
WHERE om.ORDER_INST IS NOT NULL;


-- =====================================================================
-- STEP 3: Time-ordered sequence per case (the GRU/GNN substrate).
-- One row per (LogID, order): token + position + inter-order gap.
-- Ties (same timestamp) broken by source then group for determinism.
-- =====================================================================
SELECT
    LogID,
    PAT_ID,
    EncounterCSN,
    SeqInEncounter = ROW_NUMBER() OVER (PARTITION BY LogID
                                        ORDER BY OrderTime, OrderSource, OrderGroup),
    OrderTime,
    OrderSource,
    OrderGroup,
    MinutesFromPrev = DATEDIFF(MINUTE,
                        LAG(OrderTime) OVER (PARTITION BY LogID
                                             ORDER BY OrderTime, OrderSource, OrderGroup),
                        OrderTime)
FROM #Orders
ORDER BY LogID, SeqInEncounter;


-- =====================================================================
-- Sanity check (optional): sequence length + vocabulary.
-- =====================================================================
/*
SELECT
    avg_len   = AVG(cnt * 1.0),
    max_len   = MAX(cnt),
    n_cases   = COUNT(*)
FROM (SELECT LogID, cnt = COUNT(*) FROM #Orders GROUP BY LogID) x;

SELECT OrderGroup, n = COUNT(*)
FROM #Orders GROUP BY OrderGroup ORDER BY n DESC;   -- the token vocabulary
*/
