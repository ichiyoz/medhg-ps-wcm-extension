# SQL extracts (Epic Clarity / Cogito)

T-SQL scripts that build the cohort and extract the features, graph, and
labels that feed the `medhg_ps` pipeline. They run against an **Epic Clarity**
database via Cogito (Microsoft SQL Server, 2016+ — `STRING_SPLIT` is used).

> ⚠️ **All outputs are patient-level PHI.** Run only inside your institution's
> governed environment under the appropriate IRB / data-use agreement. Nothing
> these scripts emit may be committed to this repository (see root `.gitignore`).
> `OR_Notes_Before_Discharge.sql` returns free-text clinical notes — treat as
> the highest-sensitivity output.

## Shared prerequisites

Most scripts depend on two **session-scoped temp tables** in `tempdb`. They
must be created in the **same Cogito session**, before the scripts that use
them, and after `USE Clarity;`:

| Temp table | Created by | Used by |
|---|---|---|
| `#CohortLogIDs` | `Cohort_MIS_2020_2026.sql` | Bulk features, GNN extract, OR notes |
| `#UnitTypeLookup` | `UnitTypeLookup_Temp.sql` | GNN extract (A3/A5 unit classification) |

`#CohortLogIDs` (one row per surgical case: `LogID, EncounterCSN, SurgeryDate,
…`) is the spine the whole pipeline joins to. To reload it into a fresh session
without re-sampling, `medhg_ps/scripts/gen_cohort_temptable.py` regenerates a
deterministic `INSERT` script from a saved cohort.

## Run order

```
1.  Cohort_MIS_2020_2026.sql      ->  #CohortLogIDs        (run first)
2.  UnitTypeLookup_Temp.sql       ->  #UnitTypeLookup
3a. GNN_Graph_Extract.sql         ->  A1..A5 result sets
3b. Bulk_Features_From_Cohort.sql ->  bulk_features_with_label
4.  OR_Notes_Before_Discharge.sql ->  operative/progress notes  (optional, LLM layer)
```

Save each result set to the file the pipeline expects (default
`~/Downloads/medhg_ps_data/`, override via `MEDHG_PS_*` env vars — see
`medhg_ps/config.py`). Exports may be **headerless**: the readers apply column
names positionally from the SQL-derived schemas in `config.py`, so a missing
header is fine for CSV; convert to parquet with `python -m medhg_ps.convert`.

## Files

### Core pipeline

| File | Purpose | Requires | Produces |
|---|---|---|---|
| **Cohort_MIS_2020_2026.sql** | Defines the study cohort: adult (≥18) laparoscopic/robotic general-surgery cases, 2022–2026, completed (`STATUS_C=2`), with inpatient linkage and an MIS CPT; stratified random sample (≤50k, by surgery year). | `USE Clarity` | `#CohortLogIDs` (+ result set) |
| **UnitTypeLookup_Temp.sql** | Loads the enterprise gold-standard unit classification (325 NYP units → `GNNUnitType` buckets: OR/ED/Acute/Intensive/…). Generated from `Unit_Names.xlsx` by `scripts/gen_unit_type_lookup_sql.py`. | — | `#UnitTypeLookup` |
| **GNN_Graph_Extract.sql** | Builds the heterograph extracts: **A1** encounter nodes, **A2** ENC–PROV edges (surgeons + anesthesia), **A3** ENC–UNIT edges (full ED→OR→discharge trajectory, hours-weighted), **A4** provider attributes, **A5** unit attributes. Column order matches `config.A1..A5_*_COLUMNS`. | `#CohortLogIDs`, `#UnitTypeLookup` | A1–A5 (5 result sets) |
| **Bulk_Features_From_Cohort.sql** | One row per `LogID`: 40 pre-op NSQIP-style model features (age, ASA, anesthesia, 12 labs, comorbidity/med flags, BMI, procedure counts) + identifiers + the **30-day unplanned readmission** label. Column order matches `config.BULK_FEATURES_COLUMNS` (47). | `#CohortLogIDs` | `bulk_features_with_label` |
| **OR_Notes_Before_Discharge.sql** | Operative, anesthesia, admission-summary, and post-op progress notes per case, bounded surgery-start → discharge, latest version only. Feeds the LLM plain-language risk-narrative layer. **High-sensitivity free text.** | `#CohortLogIDs` | one row per (LogID, note) |

### Reference / auxiliary (not in the main run path)

| File | Purpose | Status |
|---|---|---|
| **comorbidities_draft.sql** | Scaffold CTEs deriving NSQIP comorbidities from Clarity (ICD-10 + meds). Pasted into the per-case query, not run standalone. | **DRAFT** — ICD sets need clinical review; table names need Cogito-build confirmation. |
| **NSQIP_LogID_Crosswalk_v3.sql** | Maps NSQIP-abstracted CSV MRNs → `LogID` by following Epic patient-merge chains (`MRG_TO_PAT_ID` closure). | **Deprecated** — ~32% match ceiling; kept for the IDENTITY_ID resolution chain reference. |
| **AP_Bottleneck_Analysis.sql** | Anatomic-Pathology case-tracking turnaround analysis (accession→signout, task and test intervals). Independent of the readmission model. | Standalone utility. |

## WCM Clarity implementation notes

These scripts were developed against the **Weill Cornell Medicine (WCM)**
Clarity build, which deviates from the standard Epic data model in several
ways. The facts below are distilled from the project knowledge base
(`claude_memory/` in the source tree) — each bullet cites the note it comes
from. They are the non-obvious decisions a reviewer or re-implementer needs;
on a different Epic build, re-verify them.

**Database / session**
- WCM exposes the warehouse as the `Clarity` database; queries that mix
  `tempdb` temp tables with Clarity tables need an explicit `USE Clarity;`.
  *(`clarity-db-name`)*
- Cogito SQL Server gotchas: `PERCENTILE_CONT` is window-only (no `GROUP BY`
  form); very large `IN (...)` lists overflow (error 8623 — stage into a temp
  table); `datetime` literals are limited to 3 fractional digits.
  *(`sql-server-cogito-gotchas`)*

**Label — 30-day *unplanned* readmission** (`Bulk_Features_From_Cohort.sql`)
- `ZC_HOSP_ADMSN_TYPE` is **non-standard** at WCM; unplanned =
  `HOSP_ADMSN_TYPE_C IN (3,7,8,13,14)`. Read admission type from
  `PAT_ENC_HSP`, **not** `PAT_ENC`. *(`wcm-clarity-admission-types`)*
- The model scores **post-surgery / pre-discharge**, which determines what is
  leakage vs. legitimately available. *(`periop-readmission-model-timing`)*
- Canonical training cohort is ~43,100 MIS cases with the unplanned label.
  *(`medhg-ps-43k-cohort`)*

**Care-unit trajectory & ADT** (`GNN_Graph_Extract.sql` A3, `IndexDischarge`)
- Correct trajectory filter is `EVENT_TYPE_C IN (1,2,3)` with Discharge (2)
  used only as the `LEAD` endpoint and `EVENT_SUBTYPE_C != 2` (skip Canceled)
  — **not** the `IN (1,3,4)` pattern. *(`wcm-clarity-adt-events`)*

**Unit classification** (`UnitTypeLookup_Temp.sql`, A3/A5)
- `UnitType` comes from the **JupiterGOLD** gold-standard lookup (14 curated
  categories), date-versioned for unit reclassifications. **There is no
  "Intermediate" bucket** at WCM. *(`wcm-unit-classification`)*
- JupiterGOLD (a WCM DSS layer) is **not co-queryable** with Clarity — it is
  exported and loaded into `#UnitTypeLookup`, then JOINed.
  *(`jupitergold-reference`)*

**Provider classification** (`GNN_Graph_Extract.sql` A2/A4) — ⚠️ caveat
- The validated WCM ProviderType is a 5-bucket rule keyed on the **`PROV_TYPE`
  text**, because **`MCD_PROF_CD_C` codes do not match at WCM**; anesthesia
  roles get a role-context override. *(`wcm-provider-classification`)*
- Note the committed A2 `ProviderTypeRule` still references `MCD_PROF_CD_C` —
  treat that path as unreliable on this build and prefer the `PROV_TYPE`-text
  rule. `SPECIALTY` is **not** on `CLARITY_SER`; reach it via
  `CLARITY_SER_2.PRIMARY_DEPT_ID → CLARITY_DEP.SPECIALTY`; `ACTIVE_STATUS` is
  text. *(`wcm-clarity-schema`)*

**Comorbidities** (`comorbidities_draft.sql`, bulk features)
- ICD-10 + medication derivation rules are tiered HIGH/MOD/LOW fidelity vs.
  NSQIP chart abstraction; `FNSTATUS2` (functional status) is unreliable and
  dropped. *(`nsqip-comorbidity-derivation-fidelity`)*

**Clinical notes** (`OR_Notes_Before_Discharge.sql`)
- Text lives in `HNO_INFO` / `HNO_NOTE_TEXT`; join CSN-direct, resolve type via
  `ZC_NOTE_TYPE_IP`, handle `NOTE_CSN_ID` versioning; useful inpatient note
  types are `IP_NOTE_TYPE_C IN (19,25,27,29,32)`. *(`wcm-clarity-notes`)*

**Crosswalk** (`NSQIP_LogID_Crosswalk_v3.sql`)
- Deprecated — the MRN→LogID match ceilings at ~32%; kept only for the
  IDENTITY_ID merge-chain resolution reference. *(`nsqip-clarity-crosswalk`)*

## Notes

- **Same-session temp tables:** `#`-prefixed tables live in `tempdb` and vanish
  when the session ends — keep the cohort + lookup + extract steps in one
  connection, or reload via the `scripts/gen_*` generators.
- **Cohort window:** the sample targets 2022–2026; `Cohort_MIS_2020_2026.sql`
  is the authoritative inclusion/exclusion definition for the manuscript
  Methods.
- **Schema is code, not header:** if you change a final `SELECT`'s column order,
  update the matching `*_COLUMNS` tuple in `medhg_ps/config.py` in lockstep, or
  headerless reads will mislabel columns.
