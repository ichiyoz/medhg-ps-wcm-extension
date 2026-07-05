"""Wrapper: run an analysis script with SDoH features EXCLUDED from the model
feature set (sensitivity analysis). Drops Language + the SDOH_*_Z flags from
MODEL_FEATURE_COLUMNS in memory only; the data schema (BULK_FEATURES_COLUMNS)
is untouched so loading is unaffected. Usage: python analysis/run_nosdoh.py <script.py>
"""
import sys, runpy
import medhg_ps.config as C

SDOH = {"Language", "SDOH_Housing_Z", "SDOH_Food_Z", "SDOH_Financial_Z", "SDOH_Any_Z",
        "Race", "Ethnicity", "ZIP", "PayorCategory"}
C.MODEL_FEATURE_COLUMNS = tuple(f for f in C.MODEL_FEATURE_COLUMNS if f not in SDOH)
print(f"[nosdoh] MODEL_FEATURE_COLUMNS -> {len(C.MODEL_FEATURE_COLUMNS)} features "
      f"(SDoH dropped)", flush=True)
runpy.run_path(sys.argv[1], run_name="__main__")
